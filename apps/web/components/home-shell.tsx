"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { WarpSection } from "@/components/warp-section";
import { motion } from "@/lib/motion";

const ProjectsTimeline = dynamic(
  () => import("@/components/projects-timeline").then((m) => m.ProjectsTimeline),
  { ssr: false },
);
const AboutSection = dynamic(
  () => import("@/components/about-section").then((m) => m.AboutSection),
  { ssr: false },
);
const ContactSection = dynamic(
  () => import("@/components/contact-section").then((m) => m.ContactSection),
  { ssr: false },
);
const BlackHoleFooter = dynamic(
  () => import("@/components/black-hole-footer").then((m) => m.BlackHoleFooter),
  { ssr: false },
);

export function HomeShell() {
  const [contentReady, setContentReady] = useState(false);

  useEffect(() => {
    // Load the rest of the page shortly before intro ends.
    const leadInMs = 900;
    const loadAt = Math.max(0, motion.warp.introDuration - leadInMs);
    const timer = window.setTimeout(() => {
      setContentReady(true);
    }, loadAt);

    return () => window.clearTimeout(timer);
  }, []);

  return (
    <main id="main-content" className="bg-black">
      <WarpSection />

      {contentReady ? (
        <>
          <ProjectsTimeline />
          <AboutSection />
          <BlackHoleFooter overlay={<ContactSection overlay />} />
        </>
      ) : null}
    </main>
  );
}
