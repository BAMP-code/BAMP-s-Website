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
        // Cyberpunk 2077 / Edgerunners palette — see docs/DESIGN_DIRECTION.md
        primary: "#ece9d8",
        muted: "#86866e",
        surface: "#0a0a07",
        "surface-alt": "#111009",
        border: "#23231a",
        "card-title": "#ffffff",
        "card-body": "#cfcdba",
        "section-title-start": "#fcee0a",
        "section-title-end": "#00f0ff",
        accent: "#00f0ff",
        "accent-secondary": "#ff003c",
        "brand-core": "#fcee0a",
        hazard: "#fcee0a",
        signal: "#00ff9f",
      },
      fontFamily: {
        display: ["var(--font-display)", "Arial Narrow", "Impact", "sans-serif"],
        body: ["var(--font-body)", "var(--font-manrope)", "system-ui", "sans-serif"],
        sans: ["var(--font-body)", "var(--font-manrope)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
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
