"""Frozen text encoder for text control (context/claude/text_control.md): captions / prompts -> one vector each."""

import torch

LABSE = 'sentence-transformers/LaBSE'
TEXT_DIM = 768


def caption_mean(encoder, dataset, n=4096, seed=0):
    """Mean caption vector of n training windows of `dataset` (a NoteWindowDataset with captions; empty captions
    skipped), the center of MusicConfig.text_center."""
    import random
    rng = random.Random(seed)
    caps = []
    while len(caps) < n:
        cap = dataset[rng.randrange(len(dataset))][0][-1]
        if cap:
            caps.append(cap)
    vecs = torch.cat([encoder.encode(caps[i:i + 256]) for i in range(0, n, 256)])
    return vecs.mean(0)


class TextEncoder:
    """LaBSE (multilingual BERT-base, frozen): list of strings -> (B, 768) unit vectors, the BERT pooler output
    (= sentence-transformers' CLS + Dense + tanh), L2-normalized. An empty string -> the zero vector = 'no caption'
    (what caption dropout trains), so the model's text_proj adds nothing for it. Weights in `dtype` on `device`;
    the output is float32."""

    def __init__(self, name=LABSE, device='cpu', dtype=torch.float32, max_length=128):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.max_length = max_length
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, tokens=False, max_length=None):
        """tokens=True (T2 prefix models): (B, max_length, dim) last-layer token states, padded to max_length with
        zero rows (fixed shape for torch.compile); '' -> all zero = no caption. max_length: tokens kept per text
        (default: the encoder's; section captions are short)."""
        texts = list(texts)
        if tokens:
            return self._tokens(texts, max_length or self.max_length)
        out = torch.zeros(len(texts), self.dim, device=self.device)
        idx = [i for i, t in enumerate(texts) if t]
        if idx:
            batch = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors='pt').to(self.device)
            pooled = self.model(**batch).pooler_output.float()
            out[idx] = torch.nn.functional.normalize(pooled, dim=-1)
        return out

    def _tokens(self, texts, max_length):
        out = torch.zeros(len(texts), max_length, self.dim, device=self.device)
        idx = [i for i, t in enumerate(texts) if t]
        if idx:
            batch = self.tokenizer([texts[i] for i in idx], padding='max_length', truncation=True,
                                   max_length=max_length, return_tensors='pt').to(self.device)
            states = self.model(**batch).last_hidden_state.float()
            out[idx] = states * batch['attention_mask'][..., None].float()
        return out
