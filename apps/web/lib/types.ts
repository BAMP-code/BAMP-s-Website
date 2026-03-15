export type ProjectCategory = "cs" | "ee-me" | "drawings";
export type ProjectStatus = "completed" | "in-progress";

export type MediaAsset = {
  type: "image" | "video";
  src: string;
  alt: string;
  width: number;
  height: number;
  poster?: string;
};

export type Project = {
  id: string;
  title: string;
  status: ProjectStatus;
  description: string;
  category: ProjectCategory;
  media: MediaAsset;
  links?: { label: string; url: string }[];
};

export type AboutInfo = {
  name: string;
  headline: string;
  bio: string;
  portrait: MediaAsset;
  socials: { platform: string; url: string; label: string }[];
};

export type NavLink = {
  label: string;
  href: string;
  external?: boolean;
};
