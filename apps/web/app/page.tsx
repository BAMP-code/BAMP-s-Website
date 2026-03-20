import { WarpSection } from "@/components/warp-section";
import { ProjectsTimeline } from "@/components/projects-timeline";
import { AboutSection } from "@/components/about-section";
import { ContactSection } from "@/components/contact-section";
import { BlackHoleFooter } from "@/components/black-hole-footer";
import { ImagePreloader } from "@/components/image-preloader";

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
