"use client";

import Image from "next/image";
import { Project } from "@/lib/types";
import { useSlider } from "./use-slider";
import { SliderControls } from "./slider-controls";
import { SliderDots } from "./slider-dots";
import { OptimizedVideo } from "@/components/optimized-video";

type SliderProps = {
  projects: Project[];
  label: string;
};

export function Slider({ projects, label }: SliderProps) {
  const {
    currentIndex,
    animation,
    next,
    prev,
    goToSlide,
    handleKeyDown,
    liveRef,
  } = useSlider(projects.length);

  const project = projects[currentIndex];

  const slideAnimationClass =
    animation === "next"
      ? "animate-slide-out-left"
      : animation === "prev"
        ? "animate-slide-out-right"
        : "animate-slide-in-right";

  return (
    <section
      aria-roledescription="carousel"
      aria-label={`${label} Projects`}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      className="outline-none"
    >
      <div className="relative mb-6 flex min-h-[380px] w-full items-center justify-center overflow-visible rounded-card border border-border/70 bg-surface shadow-card-sm">
        <div
          role="group"
          aria-roledescription="slide"
          aria-label={`${currentIndex + 1} of ${projects.length}`}
          className={`relative mx-2 flex min-h-[420px] w-[90%] max-w-[700px] flex-col items-center justify-end overflow-visible rounded-card-lg border border-border/80 bg-surface-alt shadow-card transition-all duration-500 ${slideAnimationClass}`}
        >
          {/* Media */}
          <div className="flex h-[220px] w-full items-center justify-center overflow-hidden rounded-t-card border-b border-border bg-[#0a1328] shadow-media-border">
            {project.media.type === "video" ? (
              <OptimizedVideo
                src={project.media.src}
                poster={project.media.poster}
                alt={project.media.alt}
              />
            ) : (
              <Image
                src={project.media.src}
                alt={project.media.alt}
                width={project.media.width}
                height={project.media.height}
                className="max-h-full max-w-full rounded-t-card bg-[#0a1328] object-contain"
                sizes="(max-width: 800px) 90vw, 700px"
                quality={75}
              />
            )}
          </div>

          {/* Content */}
          <div className="flex w-full flex-col items-start gap-[10px] overflow-y-auto rounded-b-card bg-surface-alt px-[18px] py-5 font-sans text-[1.08rem] text-primary shadow-[0_2px_8px_rgba(0,0,0,0.15)]">
            <h3 className="text-[1.3rem] font-semibold leading-tight tracking-[0.01em] text-card-title">
              {project.title}
            </h3>
            <p className="text-[1.05rem] font-medium leading-tight tracking-[0.05em] text-accent">
              {project.status === "completed" ? "COMPLETED" : "IN PROGRESS"}
            </p>
            <p className="leading-relaxed text-card-body">
              {project.description}
            </p>
          </div>
        </div>
      </div>

      <SliderControls onPrev={prev} onNext={next} />
      <SliderDots
        total={projects.length}
        current={currentIndex}
        onSelect={goToSlide}
      />

      {/* Live region for screen reader announcements */}
      <div
        ref={liveRef}
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      />
    </section>
  );
}
