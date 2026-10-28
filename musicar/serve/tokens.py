"""
Invite tokens: one per person, shown once when made; only their SHA-256 is kept (SQLite, engine.toml tokens_db).

    python -m musicar.serve.tokens add anna        # prints anna's token (give it to her)
    python -m musicar.serve.tokens list
    python -m musicar.serve.tokens revoke anna
"""

import argparse
import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from musicar.serve.config import CONFIG, EngineConfig


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class TokenStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()  # one connection, used from the request threads
        self.db.execute('CREATE TABLE IF NOT EXISTS tokens (name TEXT PRIMARY KEY, hash TEXT UNIQUE NOT NULL, '
                        'created REAL NOT NULL, last_used REAL, jobs INTEGER NOT NULL DEFAULT 0, '
                        'revoked INTEGER NOT NULL DEFAULT 0)')

    def add(self, name):
        """A new token for `name` (replacing a revoked one); ValueError if name already has an active token."""
        token = 'mus_' + secrets.token_urlsafe(24)
        with self.lock:
            row = self.db.execute('SELECT revoked FROM tokens WHERE name = ?', (name,)).fetchone()
            if row is not None and not row[0]:
                raise ValueError(f'{name} already has a token (revoke it first)')
            self.db.execute('INSERT OR REPLACE INTO tokens (name, hash, created) VALUES (?, ?, ?)',
                            (name, digest(token), time.time()))
        return token

    def check(self, token):
        """The name of an active token, else None; notes the use."""
        with self.lock:
            row = self.db.execute('SELECT name FROM tokens WHERE hash = ? AND revoked = 0', (digest(token),)).fetchone()
            if row is not None:
                self.db.execute('UPDATE tokens SET last_used = ? WHERE name = ?', (time.time(), row[0]))
        return row and row[0]

    def count_job(self, name):
        with self.lock:
            self.db.execute('UPDATE tokens SET jobs = jobs + 1 WHERE name = ?', (name,))

    def revoke(self, name):
        with self.lock:
            return self.db.execute('UPDATE tokens SET revoked = 1 WHERE name = ? AND revoked = 0', (name,)).rowcount > 0

    def list(self):
        with self.lock:
            return self.db.execute('SELECT name, created, last_used, jobs, revoked FROM tokens ORDER BY created'
                                   ).fetchall()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['add', 'list', 'revoke'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--config', default=str(CONFIG), help='engine settings (for tokens_db)')
    args = ap.parse_args()
    store = TokenStore(EngineConfig.load(args.config).tokens_path)
    if args.action != 'list' and not args.name:
        ap.error(f'{args.action} needs a name')
    if args.action == 'add':
        print(store.add(args.name))
    elif args.action == 'revoke':
        print('revoked' if store.revoke(args.name) else f'no active token for {args.name}')
    else:
        day = lambda t: time.strftime('%Y-%m-%d %H:%M', time.localtime(t)) if t else '-'  # noqa: E731
        for name, created, used, jobs, revoked in store.list():
            print(f"{name:20s} made {day(created)}  last used {day(used):16s}  {jobs:5d} jobs"
                  + ("  REVOKED" if revoked else ""))


if __name__ == '__main__':
    main()
