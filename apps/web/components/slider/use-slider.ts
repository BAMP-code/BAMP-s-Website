"use client";

import { useState, useCallback, useRef } from "react";
import { motion } from "@/lib/motion";

export function useSlider(totalSlides: number) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [animation, setAnimation] = useState<string | null>(null);
  const isAnimating = useRef(false);
  const liveRef = useRef<HTMLDivElement>(null);

  const announce = useCallback((message: string) => {
    if (liveRef.current) {
      liveRef.current.textContent = message;
    }
  }, []);

  const goToSlide = useCallback(
    (index: number, direction: 1 | -1) => {
      if (isAnimating.current || index === currentIndex) return;
      isAnimating.current = true;

      setAnimation(direction === 1 ? "next" : "prev");

      setTimeout(() => {
        setCurrentIndex(index);
        setAnimation(null);
        isAnimating.current = false;
        announce(`Slide ${index + 1} of ${totalSlides}`);
      }, motion.slide.duration);
    },
    [currentIndex, totalSlides, announce],
  );

  const next = useCallback(() => {
    const nextIdx = (currentIndex + 1) % totalSlides;
    goToSlide(nextIdx, 1);
  }, [currentIndex, totalSlides, goToSlide]);

  const prev = useCallback(() => {
    const prevIdx = (currentIndex - 1 + totalSlides) % totalSlides;
    goToSlide(prevIdx, -1);
  }, [currentIndex, totalSlides, goToSlide]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        next();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        prev();
      }
    },
    [next, prev],
  );

  return {
    currentIndex,
    animation,
    next,
    prev,
    goToSlide,
    handleKeyDown,
    liveRef,
  };
}
