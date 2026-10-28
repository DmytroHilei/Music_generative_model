// Sign in with GitHub, step 1: off to GitHub's consent page with a one-time state (checked by the callback).

import { NextRequest, NextResponse } from 'next/server';
import { config } from '@/lib/github';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const c = config();
  if (!c.clientId || !c.clientSecret || !c.adminKey || !c.engine) {
    return NextResponse.redirect(new URL('/?signin=error&reason=off', req.url));
  }
  const state = crypto.randomUUID();
  const url = new URL(`${c.github}/login/oauth/authorize`);
  url.searchParams.set('client_id', c.clientId);
  url.searchParams.set('redirect_uri', `${req.nextUrl.origin}/api/auth/github/callback`);
  url.searchParams.set('state', state);
  const res = NextResponse.redirect(url);
  res.cookies.set('gh_state', state, {
    httpOnly: true, sameSite: 'lax', secure: req.nextUrl.protocol === 'https:', path: '/api/auth/github', maxAge: 600,
  });
  return res;
}
