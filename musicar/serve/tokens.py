"""
Invite tokens: shown once when made; only their SHA-256 is kept (SQLite, engine.toml tokens_db). A name (a person) can
hold several tokens: one per invite link, and one per browser that signs in with GitHub (the page's sign-in asks the
engine for a gh-<login> token through /v1/admin/invites, with the admin key).

    python -m musicar.serve.tokens add anna        # prints anna's token (give it to her as an invite link)
    python -m musicar.serve.tokens list
    python -m musicar.serve.tokens revoke anna     # every token of anna; GitHub sign-in can't make new ones after
    python -m musicar.serve.tokens admin-key       # makes the admin key file (for the page's sign-in), prints the key
"""

import argparse
import hashlib
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from musicar.serve.config import CONFIG, EngineConfig

NAME = re.compile(r'^[a-z0-9][a-z0-9_.-]{0,63}$')  # GitHub logins (lowercased) with the gh- prefix fit


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Revoked(Exception):
    """issue() for a name whose tokens were revoked."""


class TokenStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()  # one connection, used from the request threads
        cols = [r[1] for r in self.db.execute('PRAGMA table_info(tokens)')]
        pk = [r[1] for r in self.db.execute('PRAGMA table_info(tokens)') if r[5]]
        if cols and pk == ['name']:  # the first layout: one token per name -> one row per token
            self.db.executescript('ALTER TABLE tokens RENAME TO tokens_v1;')
        self.db.execute('CREATE TABLE IF NOT EXISTS tokens (hash TEXT PRIMARY KEY, name TEXT NOT NULL, '
                        'created REAL NOT NULL, last_used REAL, jobs INTEGER NOT NULL DEFAULT 0, '
                        'revoked INTEGER NOT NULL DEFAULT 0)')
        self.db.execute('CREATE INDEX IF NOT EXISTS tokens_name ON tokens (name)')
        if cols and pk == ['name']:
            self.db.executescript('INSERT INTO tokens SELECT hash, name, created, last_used, jobs, revoked '
                                  'FROM tokens_v1; DROP TABLE tokens_v1;')

    def _new(self, name):
        token = 'mus_' + secrets.token_urlsafe(24)
        self.db.execute('INSERT INTO tokens (hash, name, created) VALUES (?, ?, ?)', (digest(token), name, time.time()))
        return token

    def add(self, name):
        """A new invite token for `name`; ValueError if the name is malformed or already has an active token (one
        invite link per person: revoke it first to replace a lost one)."""
        if not NAME.match(name):
            raise ValueError(f'bad name {name!r}: lowercase letters, digits, . _ - (at most 64)')
        with self.lock:
            if self.db.execute('SELECT 1 FROM tokens WHERE name = ? AND revoked = 0', (name,)).fetchone():
                raise ValueError(f'{name} already has a token (revoke it first)')
            return self._new(name)

    def issue(self, name):
        """One more token for `name` (a browser signing in); Revoked if the name's tokens were revoked and none is
        active, so a revoke can't be undone by signing in again (tokens add <name> allows the name again)."""
        if not NAME.match(name):
            raise ValueError(f'bad name {name!r}')
        with self.lock:
            rows = self.db.execute('SELECT revoked FROM tokens WHERE name = ?', (name,)).fetchall()
            if rows and all(r[0] for r in rows):
                raise Revoked(name)
            return self._new(name)

    def check(self, token):
        """The name of an active token, else None; notes the use."""
        with self.lock:
            h = digest(token)
            row = self.db.execute('SELECT name FROM tokens WHERE hash = ? AND revoked = 0', (h,)).fetchone()
            if row is not None:
                self.db.execute('UPDATE tokens SET last_used = ? WHERE hash = ?', (time.time(), h))
        return row and row[0]

    def count_job(self, name):
        """One job on the name's most recently used active token."""
        with self.lock:
            self.db.execute('UPDATE tokens SET jobs = jobs + 1 WHERE hash = (SELECT hash FROM tokens WHERE name = ? '
                            'AND revoked = 0 ORDER BY last_used DESC LIMIT 1)', (name,))

    def revoke(self, name):
        """Every active token of `name`; False if there was none."""
        with self.lock:
            return self.db.execute('UPDATE tokens SET revoked = 1 WHERE name = ? AND revoked = 0', (name,)).rowcount > 0

    def list(self):
        """(name, first made, last used, jobs, active tokens, revoked tokens) per name."""
        with self.lock:
            return self.db.execute('SELECT name, MIN(created), MAX(last_used), SUM(jobs), SUM(1 - revoked), '
                                   'SUM(revoked) FROM tokens GROUP BY name ORDER BY MIN(created)').fetchall()


def make_admin_key(path):
    """A new admin key in `path` (only the owner can read it); returns it."""
    key = 'adm_' + secrets.token_urlsafe(32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600)
    path.chmod(0o600)
    path.write_text(key + '\n')
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['add', 'list', 'revoke', 'admin-key'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--config', default=str(CONFIG), help='engine settings (for tokens_db, admin_key_file)')
    args = ap.parse_args()
    config = EngineConfig.load(args.config)
    if args.action == 'admin-key':
        if config.admin_key_path.exists():
            ap.error(f'{config.admin_key_path} exists (delete it to make a new key, then update the page)')
        print(make_admin_key(config.admin_key_path))
        return
    store = TokenStore(config.tokens_path)
    if args.action != 'list' and not args.name:
        ap.error(f'{args.action} needs a name')
    if args.action == 'add':
        try:
            print(store.add(args.name))
        except ValueError as e:
            ap.error(str(e))
    elif args.action == 'revoke':
        print('revoked' if store.revoke(args.name) else f'no active token for {args.name}')
    else:
        day = lambda t: time.strftime('%Y-%m-%d %H:%M', time.localtime(t)) if t else '-'  # noqa: E731
        for name, created, used, jobs, active, revoked in store.list():
            print(f"{name:24s} made {day(created)}  last used {day(used):16s}  {jobs:5d} jobs  {active} active"
                  + (f", {revoked} revoked" if revoked else "") + ("  REVOKED" if not active else ""))


if __name__ == '__main__':
    main()
