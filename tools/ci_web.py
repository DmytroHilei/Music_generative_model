"""The CI reports of tests/ci.py (logs/ci/: index.html, <run>/report.html, report.txt, test logs) on the dashboard
(tools/status.py --web) at /ci/. Read-only; files outside logs/ci are refused (the 'latest' symlink points inside)."""

from pathlib import Path
from urllib.parse import unquote

CI_DIR = Path(__file__).resolve().parents[1] / 'logs' / 'ci'
TYPES = {'.html': 'text/html; charset=utf-8', '.json': 'application/json'}


def handle(method, path):
    """(status code, content type, body bytes) for /ci/ and below, None for other paths (token already checked by
    the caller, who also redirects /ci to /ci/ so the pages' relative links resolve)."""
    route = path.split('?', 1)[0]
    if not route.startswith('/ci/'):
        return None
    if method != 'GET':
        return 405, 'text/plain', b'read only'
    rel = unquote(route[len('/ci/'):]) or 'index.html'
    root = CI_DIR.resolve()
    f = (root / rel).resolve()
    if f.is_dir():
        f = f / 'report.html'
    if not f.is_relative_to(root) or not f.is_file():
        return 404, 'text/plain', b'not found'
    return 200, TYPES.get(f.suffix, 'text/plain; charset=utf-8'), f.read_bytes()
