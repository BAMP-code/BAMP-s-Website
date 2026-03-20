import { about } from "@/content/about";
import { LinkedInIcon, GitHubIcon, HandshakeIcon } from "@/components/social-icons";

const OUTLOOK_COMPOSE_URL =
  "https://outlook.office.com/mail/deeplink/compose?to=pineda.bamp@gmail.com&subject=Portfolio%20Inquiry%20from%20Website";

type ContactSectionProps = {
  overlay?: boolean;
};

export function ContactSection({ overlay = false }: ContactSectionProps) {
  const linkedin = about.socials.find((s) => s.platform === "LinkedIn");
  const github = about.socials.find((s) => s.platform === "GitHub");
  const handshake = about.socials.find((s) => s.platform === "Handshake");

  return (
    <section
      id="contact"
      aria-label="Contact"
      className={
        overlay
          ? "px-[clamp(16px,3.5vw,72px)]"
          : "scroll-mt-32 bg-black px-[clamp(16px,3.5vw,72px)] pb-20 pt-10"
      }
    >
      <div
        className={`mx-auto w-full rounded-3xl border p-6 sm:p-8 ${
          overlay
            ? "border-white/20 bg-black/80"
            : "border-white/15 bg-surface-alt"
        }`}
      >
        <p className="mb-2 text-xs uppercase tracking-[0.28em] text-muted">Contact</p>
        <h2 className="text-3xl font-semibold text-white sm:text-4xl">Get in touch.</h2>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted sm:text-base">
          Always happy to chat about projects, collaborations, or anything else.
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-4">
          <a
            href={OUTLOOK_COMPOSE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-accent-secondary bg-accent-secondary px-5 py-2 text-sm font-semibold text-black transition-colors hover:brightness-125"
          >
            Email Me
          </a>

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
    </section>
  );
}
