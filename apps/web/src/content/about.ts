import type { AboutInfo } from "@/lib/types";

export const about: AboutInfo = {
  name: "Bryan",
  headline: "BAMP",
  bio: "I'm a senior at Stanford University studying Computer Science on the Artificial Intelligence track. I build autonomous AI agents, ML pipelines, and full-stack systems — most recently architecting stateful memory and violation-detection infrastructure at Alterion. Outside of work, I enjoy powerlifting, playing piano, and immersing myself in video games.",
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
