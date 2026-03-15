"use client";

type SliderControlsProps = {
  onPrev: () => void;
  onNext: () => void;
};

export function SliderControls({ onPrev, onNext }: SliderControlsProps) {
  return (
    <div className="relative z-[2] mt-3 flex items-center justify-center gap-6">
      <button
        onClick={onPrev}
        aria-label="Previous slide"
        className="flex h-12 w-12 items-center justify-center rounded-full border border-border bg-surface-alt text-primary shadow-glow transition-all hover:scale-[1.03] hover:border-accent/80 hover:text-accent"
      >
        <svg
          width="18"
          height="18"
          viewBox="0 0 18 18"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M11 2L4 9L11 16"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
      <button
        onClick={onNext}
        aria-label="Next slide"
        className="flex h-12 w-12 items-center justify-center rounded-full border border-border bg-surface-alt text-primary shadow-glow transition-all hover:scale-[1.03] hover:border-accent/80 hover:text-accent"
      >
        <svg
          width="18"
          height="18"
          viewBox="0 0 18 18"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M7 2L14 9L7 16"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}
