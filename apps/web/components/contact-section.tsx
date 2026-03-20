const OUTLOOK_COMPOSE_URL =
  "https://outlook.office.com/mail/deeplink/compose?to=pineda.bamp@gmail.com&subject=Portfolio%20Inquiry%20from%20Website";

type ContactSectionProps = {
  overlay?: boolean;
};

export function ContactSection({ overlay = false }: ContactSectionProps) {
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
            ? "border-white/20 bg-black/65 backdrop-blur-md"
            : "border-white/15 bg-[#09090d]"
        }`}
      >
        <p className="mb-2 text-xs uppercase tracking-[0.28em] text-white/45">Contact</p>
        <h2 className="text-3xl font-semibold text-white sm:text-4xl">Let&apos;s build something.</h2>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-white/70 sm:text-base">
          Open to discussing internships, projects, and collaboration opportunities.
        </p>

        <div className="mt-6 flex flex-wrap gap-3">
          <a
            href="https://www.linkedin.com/in/bryan-pineda-b5464424b"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-white/25 px-5 py-2 text-sm font-medium text-white transition-colors hover:border-white hover:bg-white hover:text-black"
          >
            LinkedIn
          </a>
          <a
            href={OUTLOOK_COMPOSE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-[#ff5f48] bg-[#ff5f48] px-5 py-2 text-sm font-semibold text-black transition-colors hover:bg-[#ff826f] hover:border-[#ff826f]"
          >
            Email via Outlook
          </a>
        </div>
      </div>
    </section>
  );
}
