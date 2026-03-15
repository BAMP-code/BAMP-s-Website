"use client";

export function SkipLink() {
  return (
    <a
      href="#main-content"
      className="fixed left-2 top-2 z-50 -translate-y-full rounded bg-primary px-4 py-2 text-white transition-transform focus:translate-y-0"
    >
      Skip to content
    </a>
  );
}
