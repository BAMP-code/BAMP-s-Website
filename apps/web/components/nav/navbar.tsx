"use client";

import { useState } from "react";
import { navLinks } from "@/content/nav";
import { MobileDrawer } from "./mobile-drawer";
import { BrandMark } from "@/components/brand-mark";

export function Navbar() {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-40 overflow-hidden border-b border-border/70 bg-surface/80 shadow-nav backdrop-blur-xl">
      <ul className="mx-auto flex w-[98%] max-w-6xl list-none items-center justify-end">
        <li className="mr-auto h-[52px]">
          <a
            href="https://www.linkedin.com/in/bryan-pineda-b5464424b"
            className="flex h-full items-center gap-2 px-[30px] font-sans text-primary no-underline transition-colors hover:text-accent"
          >
            <BrandMark className="h-7 w-7 animate-orbit-glow" />
            BAMP
          </a>
        </li>

        {/* Desktop nav links */}
        {navLinks.map((link) => (
          <li key={link.href} className="hidden h-[52px] md:block">
            <a
              href={link.href}
              className="flex h-full items-center px-[30px] font-sans text-primary/90 no-underline transition-colors hover:text-accent"
              {...(link.external
                ? { target: "_blank", rel: "noopener noreferrer" }
                : {})}
            >
              {link.label}
            </a>
          </li>
        ))}

        {/* Mobile menu button */}
        <li className="block h-[52px] md:hidden">
          <button
            onClick={() => setDrawerOpen(true)}
            aria-expanded={drawerOpen}
            aria-controls="mobile-nav-drawer"
            aria-label="Open navigation menu"
            className="flex h-full items-center px-[30px] text-primary transition-colors hover:text-accent"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="26"
              viewBox="0 -960 960 960"
              width="26"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M120-240v-80h720v80H120Zm0-200v-80h720v80H120Zm0-200v-80h720v80H120Z" />
            </svg>
          </button>
        </li>
      </ul>

      <MobileDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </nav>
  );
}
