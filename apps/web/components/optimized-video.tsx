"use client";

import { useEffect, useRef, useState } from "react";

type OptimizedVideoProps = {
  src: string;
  poster?: string;
  alt: string;
  fit?: "cover" | "contain";
  unstyled?: boolean;
  autoplay?: boolean;
  loop?: boolean;
  preload?: "none" | "metadata" | "auto";
};

export function OptimizedVideo({
  src,
  poster,
  alt,
  fit = "contain",
  unstyled = false,
  autoplay = true,
  loop = true,
  preload = "none",
}: OptimizedVideoProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const fitClass = fit === "cover" ? "object-cover" : "object-contain";
  const mediaClass = unstyled
    ? `h-full w-full ${fitClass}`
    : `h-full w-full rounded-t-card bg-surface-alt ${fitClass}`;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "100px" },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="h-full w-full" aria-label={alt}>
      {isVisible ? (
        <video
          autoPlay={autoplay}
          muted
          loop={loop}
          playsInline
          preload={preload}
          poster={poster}
          className={mediaClass}
        >
          <source src={src} type="video/mp4" />
        </video>
      ) : poster ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={poster}
          alt={alt}
          className={mediaClass}
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-surface-alt text-muted">
          Loading video...
        </div>
      )}
    </div>
  );
}
