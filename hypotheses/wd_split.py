"""Split weight decay: Muon's matrices, the input embeddings and the rest of AdamW's decayed group each get their own
decay. build_muon_optimizer puts the embedding tables (note streams, program, instrument-set and density condition)
in AdamW's decayed group with the heads (musicar/training/optim.py), so a global --weight_decay moves all three.
Items 1-2 of the 2026-11-20 scaling plan: wd30x_trunk (D-R3, muon=3.0: the 30x decay on the trunk only, AdamW stays
at 0.1, against the embedding-collapse confound of wd30x) and embwd0 (D-O4, emb=0.0: no decay on the 292,096
embedding parameters at 42M; Min et al. predict it loses ~0.02).
Expected: wd30x_trunk at or below wd30x (the decay helps through the trunk); embwd0 worse than control by >= T
(0.012) keeps the embedding decay, better by >= T exempts the embeddings at XL, null keeps the default.
    python tools/sweeps/aug_sweep.py --arms wd30x_trunk,embwd0   # --hypothesis=wd_split:muon=3.0 / :emb=0.0
"""

from hypotheses import Hypothesis as Base

EMBEDDINGS = tuple(f'transformer.music_embeddings.{s}.' for s in ('pitch', 'velocity', 'duration', 'delta_time')) + (
    'transformer.program.', 'transformer.cond_inst.', 'transformer.cond_density.')


class Hypothesis(Base):
    name = 'wd_split'
    muon = None          # Muon's weight decay (the transformer-block matrices); None = --weight_decay
    emb = None           # the embedding tables' decay (their own AdamW group); None = --weight_decay
    adam = None          # the rest of AdamW's decayed group (heads, embedding projection); None = --weight_decay

    def model(self, model):
        self.net = model  # the parameter names at optimizer time (fp8 conversion swaps only the block Linears)

    def optimizer(self, optimizer):
        muon, adamw = optimizer.optimizers
        assert type(muon).__name__ == 'Muon', 'wd_split needs optimizer_name=muon'
        names = {id(p): n for n, p in self.net.named_parameters()}
        if self.muon is not None:
            for g in muon.param_groups:
                g['weight_decay'] = float(self.muon)
        if self.emb is not None or self.adam is not None:
            g0 = adamw.param_groups[0]
            assert g0['weight_decay'] > 0, 'wd_split: AdamW group 0 is not the decayed group'
            emb = [p for p in g0['params'] if names[id(p)].startswith(EMBEDDINGS)]
            assert emb, 'wd_split: no embedding in the decayed AdamW group'
            g0['params'] = [p for p in g0['params'] if not names[id(p)].startswith(EMBEDDINGS)]
            emb_wd = g0['weight_decay'] if self.emb is None else float(self.emb)
            if self.adam is not None:
                g0['weight_decay'] = float(self.adam)
            adamw.add_param_group({'params': emb, 'weight_decay': emb_wd})
        groups = [('Muon', muon.param_groups[0])] + [(f'AdamW {i}', g) for i, g in enumerate(adamw.param_groups)]
        print('wd_split: ' + ' | '.join(f"{k}: {sum(p.numel() for p in g['params']):,} params, wd {g['weight_decay']:g}"
                                        for k, g in groups), flush=True)
