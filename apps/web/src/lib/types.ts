export type MediaAsset = {
  type: "image" | "video";
  src: string;
  alt: string;
  width: number;
  height: number;
  poster?: string;
};

// Project/category types live in lib/projects.ts, inferred from the Zod
// schema that validates what Supabase actually returns.

export type AboutInfo = {
  name: string;
  headline: string;
  bio: string;
  portrait: MediaAsset;
  socials: { platform: string; url: string; label: string }[];
  skills?: string[];
  education?: string;
  resumeUrl?: string;
};

export type NavLink = {
  label: string;
  href: string;
  external?: boolean;
};
