import type { AboutInfo } from "@/lib/types";

export const about: AboutInfo = {
  name: "Bryan",
  headline: "BAMP",
  bio: "I graduated from Stanford with a CS degree, AI track. Now I'm a founding engineer at Alterion, leading the AI team. Outside of that: powerlifting, piano, and more video games than I'll admit to.",
  portrait: {
    type: "image",
    src: "/images/headshot.JPG",
    alt: "Bryan Pineda headshot",
    width: 1200,
    height: 1600,
  },
  socials: [
    {
      platform: "LinkedIn",
      url: "https://www.linkedin.com/in/bryan-pineda-b5464424b",
      label: "Bryan's LinkedIn profile",
    },
    {
      platform: "GitHub",
      url: "https://github.com/BAMP-code",
      label: "Bryan's GitHub profile",
    },
    {
      platform: "Handshake",
      url: "https://app.joinhandshake.com/profiles/5up4ru",
      label: "Bryan's Handshake profile",
    },
    {
      platform: "Email",
      url: "mailto:pineda.bamp@gmail.com",
      label: "Email Bryan",
    },
  ],
  skills: [
    "Python",
    "C/C++",
    "TypeScript",
    "PyTorch",
    "TensorFlow",
    "Docker",
    "Kubernetes",
    "LangChain",
    "React",
    "SQL",
    "AWS",
    "GCP",
  ],
  education: "Stanford University, B.S. Computer Science (AI Track)",
  resumeUrl: "/resume.pdf",
};
