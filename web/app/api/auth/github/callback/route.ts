// Sign in with GitHub, step 2: GitHub sends the visitor back here. Who are they, did the owner approve their access
// request, and if so a new engine token for this browser, handed to the page in a short-lived cookie (not the URL).
// The page reads ?signin=ok|pending|none|revoked|error (&as=login, &reason=...) and shows what happened.

import { NextRequest, NextResponse } from 'next/server';
import { SIGNIN_COOKIE, accessStatus, config, exchangeCode, login, mintToken } from '@/lib/github';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const secure = req.nextUrl.protocol === 'https:';
  const back = (status: string, extra: Record<string, string> = {}, token?: string) => {
    const url = new URL('/', req.url);
    url.searchParams.set('signin', status);
    for (const [k, v] of Object.entries(extra)) url.searchParams.set(k, v);
    const res = NextResponse.redirect(url);
    res.cookies.delete({ name: 'gh_state', path: '/api/auth/github' });
    if (token) res.cookies.set(SIGNIN_COOKIE, token, { httpOnly: false, sameSite: 'lax', secure, path: '/', maxAge: 120 });
    res.headers.set('Cache-Control', 'no-store');
    return res;
  };
  const c = config();
  const code = req.nextUrl.searchParams.get('code');
  const state = req.nextUrl.searchParams.get('state');
  if (!code || !state || state !== req.cookies.get('gh_state')?.value) return back('error', { reason: 'state' });
  let user = '';
  try {
    const ghToken = await exchangeCode(c, code, `${req.nextUrl.origin}/api/auth/github/callback`);
    user = await login(c, ghToken);
    const status = await accessStatus(c, ghToken, user);
    if (status !== 'approved') return back(status, { as: user });
    const minted = await mintToken(c, user);
    if ('revoked' in minted) return back('revoked', { as: user });
    return back('ok', { as: user }, minted.token);
  } catch (e) {
    const offline = e instanceof TypeError; // fetch could not reach the engine or GitHub
    return back('error', { reason: offline ? 'offline' : (e as Error).message.slice(0, 120), ...(user ? { as: user } : {}) });
  }
}
