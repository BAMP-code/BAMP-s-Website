import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
    "./content/**/*.{ts,tsx}",
  ],
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
        mono: ["var(--font-dm-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
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
      keyframes: {
        marquee: {
          "100%": { transform: "translateX(-3800px)" },
        },
        slideOutLeft: {
          to: { opacity: "0", transform: "translateX(-44px) scale(0.985)" },
        },
        slideInRight: {
          from: { opacity: "0", transform: "translateX(44px) scale(0.985)" },
          to: { opacity: "1", transform: "translateX(0)" },
        },
        slideOutRight: {
          to: { opacity: "0", transform: "translateX(44px) scale(0.985)" },
        },
        slideInLeft: {
          from: { opacity: "0", transform: "translateX(-44px) scale(0.985)" },
          to: { opacity: "1", transform: "translateX(0)" },
        },
        fadeIn: {
          from: { opacity: "0", transform: "translateY(5px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        orbitGlow: {
          "0%, 100%": { transform: "translate3d(0, 0, 0) scale(1)", opacity: "0.55" },
          "50%": { transform: "translate3d(0, -8px, 0) scale(1.04)", opacity: "0.95" },
        },
        panelBreath: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-3px)" },
        },
        pulseDot: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
        },
      },
      animation: {
        marquee: "marquee 60s linear infinite",
        "slide-out-left": "slideOutLeft 0.4s forwards",
        "slide-in-right": "slideInRight 0.4s forwards",
        "slide-out-right": "slideOutRight 0.4s forwards",
        "slide-in-left": "slideInLeft 0.4s forwards",
        "fade-in": "fadeIn 0.3s ease",
        "orbit-glow": "orbitGlow 5.2s ease-in-out infinite",
        "panel-breath": "panelBreath 5.6s ease-in-out infinite",
        "pulse-dot": "pulseDot 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
