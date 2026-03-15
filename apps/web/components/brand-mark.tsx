type BrandMarkProps = {
  className?: string;
  title?: string;
};

export function BrandMark({
  className = "h-7 w-7",
  title = "BAMP black hole mark",
}: BrandMarkProps) {
  return (
    <svg
      viewBox="0 0 64 64"
      role="img"
      aria-label={title}
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id="holeGlow" cx="50%" cy="50%" r="52%">
          <stop offset="0%" stopColor="#ff9d3a" stopOpacity="0.88" />
          <stop offset="44%" stopColor="#ff4f00" stopOpacity="0.52" />
          <stop offset="100%" stopColor="#ff4f00" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="ringTone" x1="18%" y1="20%" x2="86%" y2="76%">
          <stop offset="0%" stopColor="#ffc46b" />
          <stop offset="58%" stopColor="#ff5a00" />
          <stop offset="100%" stopColor="#7a1f00" />
        </linearGradient>
        <filter id="ringBlur" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="1.8" />
        </filter>
      </defs>

      <circle cx="32" cy="32" r="22" fill="url(#holeGlow)" />
      <path
        d="M11 34C11 27 17 21 24 21C30 21 35 24 39 30C43 35 46 39 51 39C55 39 58 36 58 31C58 24 52 18 44 18C37 18 32 22 28 27C24 32 21 35 16 35C13 35 11 35 11 34Z"
        fill="none"
        stroke="url(#ringTone)"
        strokeWidth="10"
        strokeLinecap="round"
        strokeLinejoin="round"
        filter="url(#ringBlur)"
        opacity="0.84"
      />
      <circle cx="32" cy="32" r="11" fill="#040404" />
    </svg>
  );
}
