import { AboutInfo } from "@/lib/types";

export const about: AboutInfo = {
  name: "Bryan",
  headline: "BAMP",
  bio: "Bryan is currently a Computer Science student at Stanford University, with a strong interest in the intersection of Artificial Intelligence and Electrical Engineering. He is passionate about exploring how emerging technologies can improve the lives of millions of people around the world. Outside of academics, Bryan enjoys powerlifting, playing piano, and immersing himself in video games.",
  portrait: {
    type: "image",
    src: "/images/me-pic.png",
    alt: "Bryan's portrait",
    width: 600,
    height: 800,
  },
  socials: [
    {
      platform: "LinkedIn",
      url: "https://www.linkedin.com/in/bryan-pineda-b5464424b",
      label: "Bryan's LinkedIn profile",
    },
    {
      platform: "Email",
      url: "mailto:pineda.bamp@gmail.com",
      label: "Email Bryan",
    },
  ],
};
