import dynamic from "next/dynamic";
import { WarpSection } from "@/components/warp-section";
import { ProjectsTimeline } from "@/components/projects-timeline";
import { AboutSection } from "@/components/about-section";
import { ContactSection } from "@/components/contact-section";
import { ImagePreloader } from "@/components/image-preloader";

// BlackHoleFooter is below the fold and ships an expensive RAF/IO/SVG
// pipeline. Code-split it so its JS isn't in the initial bundle.
const BlackHoleFooter = dynamic(
  () => import("@/components/black-hole-footer").then((m) => m.BlackHoleFooter),
);

export default function Home() {
  return (
    <main id="main-content" className="bg-black">
      <ImagePreloader />
      <WarpSection />
      <ProjectsTimeline />
      <AboutSection />
      <BlackHoleFooter overlay={<ContactSection overlay />} />
    </main>
  );
}
