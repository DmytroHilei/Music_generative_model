'use client';

// A one-line message above the page (the result of a GitHub sign-in), until dismissed.

import { useApp } from '@/lib/app';

export default function NoticeBar() {
  const { notice, dismissNotice, t } = useApp();
  if (!notice) return null;
  return (
    <div className={`notice ${notice.kind}`} role="status">
      <span>
        {notice.text}
        {notice.link && <> <a href={notice.link.href} target="_blank" rel="noreferrer">{notice.link.label}</a></>}
      </span>
      <button type="button" className="act quiet" onClick={dismissNotice} aria-label={t('Dismiss')}>×</button>
    </div>
  );
}
