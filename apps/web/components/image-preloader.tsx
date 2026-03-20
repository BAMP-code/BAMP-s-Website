"use client";

import { useEffect } from "react";
import { projects } from "@/content/projects";

/**
 * Preloads all project images in the background after page load.
 * Uses the browser's idle time to fetch images so they're cached
 * by the time the user scrolls to them.
 */
export function ImagePreloader() {
  useEffect(() => {
    const imageSrcs = projects
      .filter((p) => p.media.type === "image")
      .map((p) => p.media.src);

    // Wait for the page to finish loading, then preload images
    // during idle time so they don't compete with critical resources.
    const preload = () => {
      for (const src of imageSrcs) {
        const img = new window.Image();
        img.src = src;
      }
    };

    const w = window as Window & { requestIdleCallback?: (cb: () => void) => number; cancelIdleCallback?: (id: number) => void };
    if (w.requestIdleCallback) {
      const id = w.requestIdleCallback(preload);
      return () => w.cancelIdleCallback?.(id);
    } else {
      const id = setTimeout(preload, 2000);
      return () => clearTimeout(id);
    }
  }, []);

  return null;
}
