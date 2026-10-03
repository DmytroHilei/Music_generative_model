"""Shared helpers for the cloud run: which token stores travel, repo names on the Hugging Face Hub, file hashing."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_secrets(path=ROOT / 'cloud' / 'secrets.env'):
    """KEY=value lines of cloud/secrets.env into the environment (already-set variables win)."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition('=')
            if sep and key and value.strip() and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_secrets()

# token stores the long run reads (config/long_450m.py): train mix + the three val sets
STORES = ['gigamidi_train', 'aria_train', 'discover_train',
          'gigamidi_clean_validation', 'aria_validation', 'discover_validation']
STORE_FILES = ('meta.json', 'offsets.npy', 'tokens.u16', 'programs.u8')  # programs.u8 only in multi-instrument stores
MANIFEST = 'manifest.json'


def api():
    from huggingface_hub import HfApi
    return HfApi()  # token from HF_TOKEN or `hf auth login`


def hf_user(hf):
    return hf.whoami()['name']


def data_repo(hf, name=None):
    """Private dataset repo with the token stores; MUSIC_DATA_REPO overrides (e.g. the rehearsal repo)."""
    name = name or os.environ.get('MUSIC_DATA_REPO') or 'music-stores'
    return name if '/' in name else f'{hf_user(hf)}/{name}'


def ckpt_repos(hf, run):
    """Two private model repos used alternately, so one complete checkpoint is always on the Hub and old versions
    never pile up (deleting a whole repo frees its storage at once; squashing history doesn't, reliably)."""
    user = hf_user(hf)
    return [f'{user}/{run}-ckpt-a', f'{user}/{run}-ckpt-b']


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def store_files(cache_dir, stores):
    files = []
    for store in stores:
        d = Path(cache_dir) / store
        if not (d / 'meta.json').exists():
            raise SystemExit(f'{d}: missing or incomplete store (no meta.json)')
        files += [d / f for f in STORE_FILES if (d / f).exists()]
    return files


def build_manifest(cache_dir, stores, workers=8):
    """{relative path: {size, sha256}}; hashes are cached by (size, mtime) in cloud/.hash_cache.json."""
    cache_path = ROOT / 'cloud' / '.hash_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    files = store_files(cache_dir, stores)

    def entry(p):
        st = p.stat()
        key = f'{p.resolve()}|{st.st_size}|{st.st_mtime_ns}'
        digest = cache.get(key) or sha256_file(p)
        return str(p.relative_to(cache_dir)), key, st.st_size, digest

    manifest = {}
    with ThreadPoolExecutor(workers) as pool:
        for rel, key, size, digest in pool.map(entry, files):
            cache[key] = digest
            manifest[rel] = {'size': size, 'sha256': digest}
    cache_path.write_text(json.dumps(cache))
    return manifest


def verify(cache_dir, manifest, workers=8, full=True):
    """List of problems (empty = every file present with the right size and, with full=True, the right sha256)."""
    def check(item):
        rel, want = item
        p = Path(cache_dir) / rel
        if not p.exists():
            return f'missing {rel}'
        if p.stat().st_size != want['size']:
            return f'size {rel}: {p.stat().st_size} != {want["size"]}'
        if full and sha256_file(p) != want['sha256']:
            return f'sha256 mismatch {rel}'
        return None
    with ThreadPoolExecutor(workers) as pool:
        return [r for r in pool.map(check, manifest.items()) if r]
