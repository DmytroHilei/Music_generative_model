// Sign in with GitHub (server side, used by app/api/auth/github/*): who the visitor is, whether the owner approved
// their access request, and the engine token for them. Access requests are issues of ACCESS_REPO opened with the
// "Request access" form (label access-request); the owner approves one by adding the label approved.
//
// Environment (Vercel project settings, never NEXT_PUBLIC_):
//   GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET   the GitHub OAuth app (callback: <site>/api/auth/github/callback)
//   ENGINE_ADMIN_KEY                         the engine's admin key (python -m musicar.serve.tokens admin-key)
//   ENGINE_ADMIN_URL                         the engine for minting tokens (default NEXT_PUBLIC_ENGINE)
//   ACCESS_REPO                              owner/name of the repository with the requests (default below)
//   ACCESS_APPROVERS                         comma-separated GitHub logins whose approved label counts (default: the
//                                            repository owner)
//   GITHUB_URL, GITHUB_API                   github.com / api.github.com (other values only for tests)

export const SIGNIN_COOKIE = 'musicar_signin'; // the minted token, read once by the page (lib/app.tsx)

export function config() {
  const repo = process.env.ACCESS_REPO ?? 'DmytroHilei/Music_generative_model';
  return {
    clientId: process.env.GITHUB_CLIENT_ID ?? '',
    clientSecret: process.env.GITHUB_CLIENT_SECRET ?? '',
    adminKey: process.env.ENGINE_ADMIN_KEY ?? '',
    engine: (process.env.ENGINE_ADMIN_URL ?? process.env.NEXT_PUBLIC_ENGINE ?? '').replace(/\/$/, ''),
    repo,
    approvers: (process.env.ACCESS_APPROVERS ?? repo.split('/')[0]).split(',').map((s) => s.trim().toLowerCase()),
    github: (process.env.GITHUB_URL ?? 'https://github.com').replace(/\/$/, ''),
    api: (process.env.GITHUB_API ?? 'https://api.github.com').replace(/\/$/, ''),
  };
}

type Config = ReturnType<typeof config>;

async function api(c: Config, token: string, path: string) {
  const res = await fetch(`${c.api}${path}`, {
    headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json', 'User-Agent': 'musicar' },
    cache: 'no-store',
  });
  if (!res.ok) throw new Error(`GitHub ${path.split('?')[0]}: ${res.status}`);
  return res.json();
}

// the visitor's GitHub token for the code GitHub sent back
export async function exchangeCode(c: Config, code: string, redirectUri: string): Promise<string> {
  const res = await fetch(`${c.github}/login/oauth/access_token`, {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ client_id: c.clientId, client_secret: c.clientSecret, code, redirect_uri: redirectUri }),
    cache: 'no-store',
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body.access_token) throw new Error(`GitHub sign-in: ${body.error ?? res.status}`);
  return body.access_token;
}

export async function login(c: Config, token: string): Promise<string> {
  return (await api(c, token, '/user')).login;
}

// 'approved' when one of the visitor's access requests got the approved label from an approver (the issue's events
// say who added it, so a label alone isn't trusted), 'pending' with requests but no such approval, 'none' without any
export async function accessStatus(c: Config, token: string, user: string): Promise<'approved' | 'pending' | 'none'> {
  const q = `repo:${c.repo} is:issue author:${user} label:access-request`;
  const found = await api(c, token, `/search/issues?q=${encodeURIComponent(q)}&per_page=20`);
  const items: { number: number; labels: { name: string }[] }[] = found.items ?? [];
  if (!items.length) return 'none';
  for (const issue of items) {
    if (!issue.labels.some((l) => l.name === 'approved')) continue;
    const events: { event: string; label?: { name: string }; actor?: { login: string } }[] =
      await api(c, token, `/repos/${c.repo}/issues/${issue.number}/events?per_page=100`);
    const by = events.filter((e) => e.event === 'labeled' && e.label?.name === 'approved').pop()?.actor?.login;
    if (by && c.approvers.includes(by.toLowerCase())) return 'approved';
  }
  return 'pending';
}

// a new engine token for this GitHub user ('revoked' if the owner revoked them)
export async function mintToken(c: Config, user: string): Promise<{ token: string } | { revoked: true }> {
  const res = await fetch(`${c.engine}/v1/admin/invites`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Admin-Key': c.adminKey },
    body: JSON.stringify({ name: `gh-${user.toLowerCase()}` }),
    cache: 'no-store',
  });
  if (res.status === 403) {
    const detail = (await res.json().catch(() => ({}))).detail ?? '';
    if (/revoked/.test(detail)) return { revoked: true };
    throw new Error('the engine refused the page\'s admin key');
  }
  if (!res.ok) throw new Error(`the engine answered ${res.status}`);
  return { token: (await res.json()).token };
}
