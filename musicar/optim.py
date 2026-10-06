"""
Muon optimizer (Keller Jordan, 2024) with the Moonlight update scaling ("Muon is Scalable for LLM Training", 2025),
plus a tiny wrapper that steps Muon (hidden matrices) and AdamW (everything else) as one optimizer.

Muon: momentum SGD whose update for a 2D weight is orthogonalized with a Newton-Schulz iteration (~U V^T of the
momentum), so all directions of the matrix get updated at a similar rate. Moonlight scaling multiplies the
orthogonalized update by 0.2 * sqrt(max(rows, cols)) so its RMS matches AdamW's: then Muon can use the SAME learning
rate and decoupled weight decay as AdamW, and the existing LR schedule works unchanged.
"""

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate the orthogonal polar factor U V^T of G with a quintic Newton-Schulz iteration (in bf16)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=6e-4, momentum=0.95, nesterov=True, weight_decay=0.1, ns_steps=5,
                 momentum_dtype=None, compile_ns=False, ns_chunk=8):
        # momentum_dtype: torch.bfloat16 halves the optimizer state (2 instead of 4 bytes per param). The buffer is a
        # smoothed direction that Newton-Schulz orthogonalizes in bf16 anyway; the fp32 weights still take the update.
        # Same-shaped matrices (qkv, attn proj, fc, mlp proj of every layer) run Newton-Schulz together as one batched
        # matmul, ns_chunk matrices at a time (caps the temporary memory: all 56 MLP matrices of a 450M model at once
        # ran out of memory on 8 GB); compile_ns=True also compiles it (one graph per shape).
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.momentum_dtype = momentum_dtype
        self.ns_chunk = ns_chunk
        self.ns = torch.compile(zeropower_via_newtonschulz5) if compile_ns else zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            by_shape = {}
            for p in group['params']:
                if p.grad is not None:
                    by_shape.setdefault(p.shape, []).append(p)
            chunks = [(shape, ps[c:c + self.ns_chunk]) for shape, ps in by_shape.items()
                      for c in range(0, len(ps), self.ns_chunk)]
            for shape, params in chunks:
                bufs = []
                for p in params:
                    state = self.state[p]
                    dtype = self.momentum_dtype or p.grad.dtype
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p.grad, dtype=dtype)
                    elif state['momentum_buffer'].dtype != dtype:  # load_state_dict casts state to the param dtype
                        state['momentum_buffer'] = state['momentum_buffer'].to(dtype)
                    bufs.append(state['momentum_buffer'])
                grads = [p.grad for p in params]
                torch._foreach_mul_(bufs, group['momentum'])
                torch._foreach_add_(bufs, grads)
                # stacked straight into bf16 (what Newton-Schulz runs in), one chunk of a shape group at a time
                g = torch.stack([(gr.add(buf, alpha=group['momentum']) if group['nesterov'] else buf).bfloat16()
                                 for gr, buf in zip(grads, bufs)])
                update = self.ns(g, steps=group['ns_steps'])
                scale = 0.2 * max(shape[0], shape[1]) ** 0.5  # Moonlight: match AdamW update RMS
                torch._foreach_mul_(params, 1 - group['lr'] * group['weight_decay'])
                torch._foreach_add_(params, list(update.unbind(0)), alpha=-group['lr'] * scale)


class CombinedOptimizer:
    """Several optimizers stepped together; exposes param_groups so the LR schedule in train.py just works."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, states):
        for opt, s in zip(self.optimizers, states):
            opt.load_state_dict(s)


def build_muon_optimizer(model, weight_decay, learning_rate, betas, device_type, momentum=0.95, style_lr_mult=1.0,
                         momentum_dtype=None, compile_ns=False):
    """Muon for the 2D matrices inside the transformer blocks, AdamW for embeddings, heads, norms.
    The style table (if any) gets its own AdamW group: lr x style_lr_mult (the train loop applies each group's
    'lr_mult'), no weight decay, so a zero-initialized conditioning vector can move far in a short fine-tune."""
    import inspect
    muon_params, adam_decay, adam_nodecay, style_params = [], [], [], []
    muon_ids = {id(p) for p in model.transformer.h.parameters() if p.dim() == 2}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in muon_ids:
            muon_params.append(p)
        elif n.startswith('transformer.style.'):
            style_params.append(p)
        elif p.dim() >= 2 and not (model.config.cascade_residual and 'cond_emb' in n):
            adam_decay.append(p)
        else:
            adam_nodecay.append(p)
    fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
    groups = [{'params': adam_decay, 'weight_decay': weight_decay},
              {'params': adam_nodecay, 'weight_decay': 0.0}]
    if style_params:
        groups.append({'params': style_params, 'weight_decay': 0.0, 'lr_mult': style_lr_mult})
    adamw = torch.optim.AdamW(groups,
                              lr=learning_rate, betas=betas, **(dict(fused=True) if fused else {}))
    muon = Muon(muon_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                momentum_dtype=momentum_dtype, compile_ns=compile_ns)
    print(f"Muon: {len(muon_params)} matrices, {sum(p.numel() for p in muon_params):,} params | "
          f"AdamW: {sum(p.numel() for p in adam_decay + adam_nodecay):,} params"
          + (f" | style table: {sum(p.numel() for p in style_params):,} params, lr x{style_lr_mult:g}" if style_params else ""))
    return CombinedOptimizer(muon, adamw)
