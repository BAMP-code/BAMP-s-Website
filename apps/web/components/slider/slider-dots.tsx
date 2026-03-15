"use client";

type SliderDotsProps = {
  total: number;
  current: number;
  onSelect: (index: number, direction: 1 | -1) => void;
};

export function SliderDots({ total, current, onSelect }: SliderDotsProps) {
  return (
    <div
      role="tablist"
      aria-label="Slide navigation"
      className="mt-[18px] flex items-center justify-center gap-[10px]"
    >
      {Array.from({ length: total }, (_, i) => (
        <button
          key={i}
          role="tab"
          aria-selected={i === current}
          aria-label={`Go to slide ${i + 1}`}
          onClick={() => onSelect(i, i > current ? 1 : -1)}
          className={`h-3 w-3 cursor-pointer rounded-full transition-colors ${
            i === current ? "bg-accent" : "bg-border"
          }`}
        />
      ))}
    </div>
  );
}
