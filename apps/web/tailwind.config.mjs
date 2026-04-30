/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,ts,tsx,js,jsx,html}"],
  theme: {
    screens: {
      xs: "400px",
      sm: "600px",
      md: "800px",
      lg: "992px",
    },
    extend: {
      colors: {
        primary: "#f4f6ff",
        muted: "#9ba4c7",
        surface: "#060608",
        "surface-alt": "#0d0d12",
        border: "#28283c",
        "card-title": "#ffffff",
        "card-body": "#d4d8ef",
        "section-title-start": "#f6ea2a",
        "section-title-end": "#00f6ff",
        accent: "#00f6ff",
        "accent-secondary": "#ff2e2e",
        "brand-core": "#f6ea2a",
      },
      fontFamily: {
        sans: ["var(--font-manrope)", "Inter", "Arial", "sans-serif"],
        serif: ["var(--font-manrope)", "Inter", "Arial", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "monospace"],
      },
      borderRadius: {
        card: "20px",
        "card-lg": "28px",
        chat: "16px",
        pill: "9999px",
      },
      boxShadow: {
        card: "0 24px 50px rgba(0, 0, 0, 0.62)",
        "card-sm": "0 12px 30px rgba(0, 0, 0, 0.45)",
        nav: "0 6px 30px rgba(0, 0, 0, 0.55)",
        chat: "0 18px 45px rgba(0, 0, 0, 0.65)",
        "media-border": "0 10px 24px rgba(0, 0, 0, 0.55)",
        glow: "0 0 0 1px rgba(0, 246, 255, 0.24), 0 0 30px rgba(255, 46, 46, 0.2)",
      },
    },
  },
  plugins: [],
};
