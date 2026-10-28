// Stroke icons drawn for musicar (16 px grid unless noted); decorative, so hidden from screen readers.

type P = { size?: number };
const svg = (size: number, children: React.ReactNode, box = 16, fill = false) => (
  <svg width={size} height={size} viewBox={`0 0 ${box} ${box}`} aria-hidden="true" fill={fill ? 'currentColor' : 'none'}
    stroke={fill ? 'none' : 'currentColor'} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    {children}
  </svg>
);

export function Logo({ size = 24 }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="1" y="1" width="22" height="22" rx="7" fill="var(--acc)" />
      <rect x="5" y="7" width="8" height="2.6" rx="1.3" fill="var(--on-acc)" />
      <rect x="10" y="11" width="9" height="2.6" rx="1.3" fill="var(--on-acc)" />
      <rect x="6" y="15" width="6" height="2.6" rx="1.3" fill="var(--on-acc)" />
    </svg>
  );
}

export const Plus = ({ size = 16 }: P) => svg(size, <path d="M8 3v10M3 8h10" />);
export const HomeIcon = ({ size = 16 }: P) => svg(size, <path d="M2.5 7.2L8 2.5l5.5 4.7V13a.5.5 0 0 1-.5.5H10V9.5H6v4H3a.5.5 0 0 1-.5-.5z" />);
export const Note = ({ size = 16 }: P) => svg(size, <><path d="M6 12.5V3.5l7-1.5v9" /><circle cx="4.5" cy="12.5" r="1.8" /><circle cx="11.5" cy="11" r="1.8" /></>);
export const Sun = ({ size = 16 }: P) => svg(size, <><circle cx="8" cy="8" r="3" /><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" /></>);
export const Moon = ({ size = 16 }: P) => svg(size, <path d="M13.5 9.6A5.6 5.6 0 0 1 6.4 2.5a5.6 5.6 0 1 0 7.1 7.1z" />);
export const Globe = ({ size = 16 }: P) => svg(size, <><circle cx="8" cy="8" r="6" /><path d="M2 8h12M8 2c1.7 1.7 2.5 3.7 2.5 6S9.7 12.3 8 14C6.3 12.3 5.5 10.3 5.5 8S6.3 3.7 8 2z" /></>);
export const Gear =({ size = 16 }: P) => svg(size, <><circle cx="8" cy="8" r="2.2" /><path d="M8 1.8v1.8M8 12.4v1.8M1.8 8h1.8M12.4 8h1.8M3.6 3.6l1.3 1.3M11.1 11.1l1.3 1.3M3.6 12.4l1.3-1.3M11.1 4.9l1.3-1.3" /></>);
export const Clock = ({ size = 14 }: P) => svg(size, <><circle cx="8" cy="8" r="6" /><path d="M8 4.5V8l2.3 1.6" /></>);
export const Chevron = ({ size = 12 }: P) => svg(size, <path d="M4 6l4 4 4-4" />);
export const Up = ({ size = 18 }: P) => svg(size, <path d="M8 13V3M3.5 7.5L8 3l4.5 4.5" />);
export const Redo = ({ size = 14 }: P) => svg(size, <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v2.8h-2.8" />);
export const Arrow = ({ size = 14 }: P) => svg(size, <path d="M3 8h10M9 4l4 4-4 4" />);
export const Pencil = ({ size = 14 }: P) => svg(size, <path d="M10.5 2.5l3 3L6 13H3v-3z" />);
export const Download = ({ size = 14 }: P) => svg(size, <path d="M8 2.5v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10" />);
export const Trash = ({ size = 14 }: P) => svg(size, <path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5" />);
export const Menu = ({ size = 18 }: P) => svg(size, <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />);
export const Play = ({ size = 16 }: P) => svg(size, <path d="M4.5 2.6v10.8c0 .5.6.8 1 .5l8.2-5.4a.6.6 0 0 0 0-1L5.5 2.1c-.4-.3-1 0-1 .5z" />, 16, true);
export const Pause = ({ size = 16 }: P) => svg(size, <><rect x="3" y="2.5" width="3.6" height="11" rx="1" /><rect x="9.4" y="2.5" width="3.6" height="11" rx="1" /></>, 16, true);

export function Dice({ size = 20, turn = 0 }: P & { turn?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden="true" style={{ transform: `rotate(${turn}deg)` }}>
      <rect x="2.5" y="2.5" width="15" height="15" rx="4" fill="none" stroke="currentColor" strokeWidth="1.6" />
      {[[7, 7], [13, 13], [13, 7], [7, 13], [10, 10]].map(([x, y]) => <circle key={`${x}${y}`} cx={x} cy={y} r="1.4" fill="currentColor" />)}
    </svg>
  );
}
