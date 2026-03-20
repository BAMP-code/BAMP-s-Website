"use client";

import { useCallback, useEffect } from "react";
import FocusTrap from "focus-trap-react";
import { navLinks } from "@/content/nav";

type MobileDrawerProps = {
  open: boolean;
  onClose: () => void;
};

export function MobileDrawer({ open, onClose }: MobileDrawerProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (open) {
      document.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
      return () => {
        document.removeEventListener("keydown", handleKeyDown);
        document.body.style.overflow = "";
      };
    }
  }, [open, handleKeyDown]);

  if (!open) return null;

  return (
    <FocusTrap>
      <div
        id="mobile-nav-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
        className="fixed inset-y-0 right-0 z-50 flex w-[260px] flex-col border-l border-border bg-surface/95 shadow-[-10px_0_24px_rgba(0,0,0,0.35)] backdrop-blur-[14px] xs:w-full"
      >
        <button
          onClick={onClose}
          aria-label="Close navigation menu"
          className="mt-5 w-full px-[30px] py-3 text-left text-primary"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="26"
            viewBox="0 -960 960 960"
            width="26"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="m256-200-56-56 224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z" />
          </svg>
        </button>
        <ul className="list-none">
          {navLinks.map((link) => (
            <li key={link.href} className="mt-5 w-full">
              <a
                href={link.href}
                onClick={onClose}
                className="block w-full px-[30px] py-2 font-sans text-primary no-underline transition-colors hover:text-accent"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </FocusTrap>
  );
}
