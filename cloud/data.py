"""
Move the token stores between the laptop and the rented GPU through a private Hugging Face dataset repo.

  laptop:   python cloud/data.py upload       # once, before renting: hashes, then a resumable upload (~41 GB)
  instance: python cloud/data.py download     # cloud/setup_instance.sh runs this and then `verify`
            python cloud/data.py verify       # every file present, right size, right sha256

The repo holds data/cache/<store>/... for the stores in cloud/hub.py plus manifest.json (size + sha256 per file).
The upload goes through a staging folder of symlinks (data/hf_upload), so nothing is copied. One commit per store;
a re-run skips stores already on the Hub with the right file sizes, and the Xet backend deduplicates chunks it
already has, so an interrupted 17 GB file doesn't start from zero. manifest.json goes last.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hub  # noqa: E402


def upload(args):
    hf = hub.api()
    repo = hub.data_repo(hf, args.repo)
    stores = args.stores.split(',')
    print(f'hashing {len(stores)} stores in {args.cache_dir} (cached after the first run) ...', flush=True)
    manifest = hub.build_manifest(args.cache_dir, stores)
    total = sum(e['size'] for e in manifest.values())
    print(f'{len(manifest)} files, {total / 1e9:.1f} GB -> {repo} (private dataset)', flush=True)

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    for rel in manifest:
        link = staging / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        target = (Path(args.cache_dir) / rel).resolve()
        if link.is_symlink() and link.resolve() == target:
            continue
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    (staging / hub.MANIFEST).write_text(json.dumps(manifest, indent=1))

    hf.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)

    def remote_sizes():
        return {f.path: f.size for f in hf.list_repo_tree(repo, repo_type='dataset', recursive=True)
                if getattr(f, 'size', None) is not None}

    for store in stores:
        files = {rel: e for rel, e in manifest.items() if rel.split('/')[0] == store}
        remote = remote_sizes()
        if all(remote.get(rel) == e['size'] for rel, e in files.items()):
            print(f'{store}: already on the Hub', flush=True)
            continue
        size = sum(e['size'] for e in files.values()) / 1e9
        for attempt in range(1, 4):
            print(f'{store}: uploading {size:.1f} GB (attempt {attempt})', flush=True)
            try:
                hf.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(staging / store),
                                 path_in_repo=store, commit_message=f'{store}')
                break
            except Exception as e:  # network hiccup: retry (Xet resends only missing chunks)
                print(f'  failed: {type(e).__name__}: {str(e)[:200]}', flush=True)
                if attempt == 3:
                    raise SystemExit(f'{store}: upload failed 3 times; re-run the same command to continue')
    hf.upload_file(path_or_fileobj=str(staging / hub.MANIFEST), path_in_repo=hub.MANIFEST, repo_id=repo,
                   repo_type='dataset', commit_message='manifest')
    remote = remote_sizes()
    bad = [rel for rel, e in manifest.items() if remote.get(rel) != e['size']]
    if bad or hub.MANIFEST not in remote:
        raise SystemExit(f'upload incomplete: {len(bad)} files missing or wrong size, e.g. {bad[:3]}; re-run')
    print(f'upload verified: all {len(manifest)} files on {repo}')


def download(args):
    from huggingface_hub import snapshot_download
    hf = hub.api()
    repo = hub.data_repo(hf, args.repo)
    print(f'downloading {repo} -> {args.cache_dir}', flush=True)
    snapshot_download(repo, repo_type='dataset', local_dir=args.cache_dir, max_workers=args.workers)
    print('download done')


def verify(args):
    manifest_path = Path(args.cache_dir) / hub.MANIFEST
    if not manifest_path.exists():
        raise SystemExit(f'{manifest_path} missing: run `python cloud/data.py download` first')
    manifest = json.loads(manifest_path.read_text())
    problems = hub.verify(args.cache_dir, manifest, workers=args.workers, full=not args.quick)
    total = sum(e['size'] for e in manifest.values())
    if problems:
        print('\n'.join(problems[:20]))
        raise SystemExit(f'data verification FAILED: {len(problems)} problems')
    print(f'data verified: {len(manifest)} files, {total / 1e9:.1f} GB' + (' (sizes only)' if args.quick else ''))
    free = shutil.disk_usage(args.cache_dir).free / 1e9
    print(f'free disk after the data: {free:.0f} GB')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('command', choices=['upload', 'download', 'verify'])
    p.add_argument('--cache-dir', default=str(hub.ROOT / 'data' / 'cache'))
    p.add_argument('--repo', default=None, help='dataset repo (default <hf user>/music-stores or $MUSIC_DATA_REPO)')
    p.add_argument('--stores', default=','.join(hub.STORES))
    p.add_argument('--staging', default=str(hub.ROOT / 'data' / 'hf_upload'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--quick', action='store_true', help='verify: sizes only, no sha256')
    args = p.parse_args()
    {'upload': upload, 'download': download, 'verify': verify}[args.command](args)


if __name__ == '__main__':
    main()
