"use client";

import { useEffect, useRef, useState } from "react";

type OptimizedVideoProps = {
  src: string;
  poster?: string;
  alt: string;
};

export function OptimizedVideo({ src, poster, alt }: OptimizedVideoProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

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
          autoPlay
          muted
          loop
          playsInline
          preload="none"
          poster={poster}
          className="max-h-full max-w-full rounded-t-card bg-surface-alt object-contain"
        >
          <source src={src} type="video/mp4" />
        </video>
      ) : poster ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={poster}
          alt={alt}
          className="max-h-full max-w-full rounded-t-card bg-surface-alt object-contain"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-surface-alt text-muted">
          Loading video...
        </div>
      )}
    </div>
  );
}
