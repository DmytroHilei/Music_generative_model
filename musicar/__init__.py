"""Core library: model, optimizer, data loading and tokenization, MIDI I/O, job status. The released models
(musicar/hub.py):

    import musicar
    model = musicar.from_pretrained('musicar-S')    # musicar-L, musicar-S, musicar-S-piano, musicar-L-caption
    musicar.generate(model, n_notes=500, seed=0).write('x.mid')
"""

__all__ = ['from_pretrained', 'generate', 'Generator']


def __getattr__(name):
    """from_pretrained / generate / Generator, imported on first use, so `import musicar` stays light (no torch)."""
    if name in ('from_pretrained', 'generate'):
        from musicar import hub
        return getattr(hub, name)
    if name == 'Generator':
        from musicar.generation.piece import Generator
        return Generator
    raise AttributeError(f"module 'musicar' has no attribute {name!r}")
