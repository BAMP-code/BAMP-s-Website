import { WarpSection } from "@/components/warp-section";
import { AboutSection } from "@/components/about-section";
import { ContactSection } from "@/components/contact-section";

export default function Home() {
  return (
    <main id="main-content" className="bg-black">
      <WarpSection />
      <AboutSection />
      <ContactSection />
    </main>
  );
}
