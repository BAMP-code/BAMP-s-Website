"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "@/lib/motion";

type SectionTitleProps = {
  children: React.ReactNode;
};

export function SectionTitle({ children }: SectionTitleProps) {
  const ref = useRef<HTMLHeadingElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setVisible(true);
      },
      { threshold: motion.sectionReveal.threshold },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <h2
      ref={ref}
      className={`section-title-gradient mb-10 text-center font-sans text-[2.4rem] font-extrabold tracking-[0.01em] transition-all sm:text-[2.8rem] ${
        visible
          ? "translate-y-0 opacity-100"
          : "translate-y-6 opacity-0"
      }`}
      style={{ transitionDuration: `${motion.sectionReveal.duration}ms` }}
    >
      {children}
    </h2>
  );
}
