"use client";

import { about } from "@/content/about";
import { useEffect, useRef, useState } from "react";
import { motion } from "@/lib/motion";

export function AboutSection() {
  const ref = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
        }
      },
      { threshold: motion.aboutReveal.threshold },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section
      id="about"
      ref={ref}
      className={`relative z-10 mx-auto mt-10 w-[min(96vw,1800px)] scroll-mt-32 rounded-[34px] px-[clamp(18px,2vw,44px)] py-8 transition-all duration-700 sm:py-10 ${
        visible ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"
      } about-surface animate-panel-breath`}
    >
      <span className="orbit-line left-8 top-8 h-8 w-14 sm:h-10 sm:w-20" />
      <span className="orbit-line bottom-9 right-10 h-6 w-16 opacity-60 sm:h-8 sm:w-24" />

      <h2 className="section-title-gradient text-3xl font-extrabold tracking-[0.26em] sm:text-5xl">
        ABOUT
      </h2>
      <p className="mt-6 max-w-4xl text-base leading-relaxed text-primary/90 sm:text-lg">
        {about.bio}
      </p>
    </section>
  );
}
