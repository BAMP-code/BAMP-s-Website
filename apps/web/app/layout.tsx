import type { Metadata } from "next";
import { Manrope } from "next/font/google";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
  display: "swap",
});

export const metadata: Metadata = {
  title: "BAMP — Bryan Pineda",
  description:
    "Bryan Pineda's portfolio — Stanford CS, embedded systems, AI, and intelligent products.",
  metadataBase: new URL("https://bamp.codes"),
  openGraph: {
    title: "BAMP — Bryan Pineda",
    description:
      "Stanford CS student building embedded systems and intelligent products.",
    url: "https://bamp.codes",
    siteName: "BAMP",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "BAMP — Bryan Pineda",
    description:
      "Stanford CS student building embedded systems and intelligent products.",
  },
  icons: { icon: "/favicon.ico" },
  other: { "theme-color": "#060608" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={manrope.variable}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
