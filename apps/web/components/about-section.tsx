import Image from "next/image";
import { about } from "@/content/about";
import {
  LinkedInIcon,
  GitHubIcon,
  HandshakeIcon,
  GraduationCapIcon,
} from "@/components/social-icons";

export function AboutSection() {
  const linkedin = about.socials.find((s) => s.platform === "LinkedIn");
  const github = about.socials.find((s) => s.platform === "GitHub");
  const handshake = about.socials.find((s) => s.platform === "Handshake");

  return (
    <section
      id="about"
      aria-label="About Bryan"
      className="relative z-10 mx-auto mt-10 w-[min(96vw,1800px)] scroll-mt-32 rounded-[34px] px-[clamp(18px,2vw,44px)] py-8 sm:py-10 about-surface animate-panel-breath"
    >
      <span className="orbit-line left-8 top-8 h-8 w-14 sm:h-10 sm:w-20" />
      <span className="orbit-line bottom-9 right-10 h-6 w-16 opacity-60 sm:h-8 sm:w-24" />

      <h2 className="section-title-gradient text-3xl font-extrabold tracking-[0.26em] sm:text-5xl">
        ABOUT
      </h2>

      <div className="mt-8 flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Headshot */}
        <div className="w-48 shrink-0 sm:w-56 lg:w-64">
          <Image
            src={about.portrait.src}
            alt={about.portrait.alt}
            width={about.portrait.width}
            height={about.portrait.height}
            className="rounded-2xl object-cover"
            sizes="(max-width: 992px) 224px, 256px"
            quality={80}
          />
        </div>

        {/* Content */}
        <div className="flex-1 space-y-5">
          <p className="max-w-3xl text-base leading-relaxed text-primary/90 sm:text-lg">
            {about.bio}
          </p>

          {/* Education */}
          {about.education && (
            <p className="flex items-center gap-2 text-sm text-muted">
              <GraduationCapIcon />
              {about.education}
            </p>
          )}

          {/* Skills */}
          {about.skills && about.skills.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {about.skills.map((skill) => (
                <span
                  key={skill}
                  className="rounded-full border border-border bg-white/5 px-3 py-1 text-xs text-card-body"
                >
                  {skill}
                </span>
              ))}
            </div>
          )}

          {/* Actions: Resume + Social icons */}
          <div className="flex flex-wrap items-center gap-4 pt-2">
            {about.resumeUrl && (
              <a
                href={about.resumeUrl}
                className="rounded-full border border-accent/30 px-5 py-2 text-sm font-medium text-accent/80 transition-colors hover:border-accent hover:text-accent"
                download
              >
                Download Resume
              </a>
            )}

            <div className="flex items-center gap-3">
              {linkedin && (
                <a
                  href={linkedin.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={linkedin.label}
                  className="text-muted transition-colors hover:text-white"
                >
                  <LinkedInIcon />
                </a>
              )}
              {github && (
                <a
                  href={github.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={github.label}
                  className="text-muted transition-colors hover:text-white"
                >
                  <GitHubIcon />
                </a>
              )}
              {handshake && (
                <a
                  href={handshake.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={handshake.label}
                  className="flex items-center gap-1.5 text-muted transition-colors hover:text-white"
                >
                  <HandshakeIcon />
                  <span className="text-xs">Handshake</span>
                </a>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
