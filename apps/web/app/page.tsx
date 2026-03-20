import { BlackHoleHero } from "@/components/black-hole-hero";
import { AboutSection } from "@/components/about-section";
import { ProjectsTimeline } from "@/components/projects-timeline";
import { ContactSection } from "@/components/contact-section";

export default function Home() {
  return (
    <main id="main-content" className="bg-black">
      <BlackHoleHero />
      <ProjectsTimeline />
      <ContactSection />
      <AboutSection />
    </main>
  );
}
