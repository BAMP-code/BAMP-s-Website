"use client";

export function SkipLink() {
  return (
    <a
      href="#main-content"
      className="fixed left-2 top-2 z-50 -translate-y-full rounded bg-accent px-4 py-2 font-semibold text-black transition-transform focus:translate-y-0"
    >
      Skip to content
    </a>
  );
}
