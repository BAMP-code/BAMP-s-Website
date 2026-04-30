import type { Project } from "@/lib/types";

export const projects: Project[] = [
  {
    id: "waypoint-prediction",
    title: "End-To-End Waypoint Prediction",
    status: "completed",
    description:
      "A deep learning project for predicting waypoints in autonomous navigation using the nuScenes dataset.",
    category: "cs",
    media: {
      type: "image",
      src: "/images/waypoint_prediction.jpg",
      alt: "End-To-End Waypoint Prediction",
      width: 700,
      height: 400,
    },
  },
  {
    id: "bio-printing-ui",
    title: "Bio Printing User Interface",
    status: "completed",
    description:
      "Developing a friendly user interface where users can view and interact with 3D documents of human vessels.",
    category: "cs",
    media: {
      type: "image",
      src: "/images/GUI.jpg",
      alt: "Bio Printing User Interface",
      width: 700,
      height: 400,
    },
  },
  {
    id: "website-for-her",
    title: "Website for Her",
    status: "completed",
    description:
      "A website created for the sole purpose of asking out my girlfriend, Daniela, on a date.",
    category: "cs",
    media: {
      type: "image",
      src: "/images/for-her.jpg",
      alt: "Website for Her",
      width: 700,
      height: 400,
    },
  },
  {
    id: "link-app",
    title: "L'Ink App",
    status: "in-progress",
    description:
      "A mobile app for collaborative note-taking, sketching, and sharing.",
    category: "cs",
    media: {
      type: "video",
      src: "/videos/link-app-demo.mp4",
      alt: "L'Ink App demo",
      width: 390,
      height: 844,
    },
  },
  {
    id: "ecg",
    title: "Electro Cardiogram",
    status: "completed",
    description:
      "A project focused on building and analyzing an ECG circuit for biomedical applications.",
    category: "ee-me",
    media: {
      type: "image",
      src: "/images/ECG_project.jpg",
      alt: "Electro Cardiogram Project",
      width: 700,
      height: 400,
    },
  },
  {
    id: "truss",
    title: "Truss",
    status: "completed",
    description:
      "Mechanical engineering project involving the design and analysis of a truss structure.",
    category: "ee-me",
    media: {
      type: "image",
      src: "/images/Truss_project.jpg",
      alt: "Truss Project",
      width: 700,
      height: 400,
    },
  },
  {
    id: "led-board",
    title: "LED Board",
    status: "completed",
    description:
      "Designed and built a custom LED board for interactive displays, we are able to play ping pong on it.",
    category: "ee-me",
    media: {
      type: "image",
      src: "/images/LED_board.jpg",
      alt: "LED Board Project",
      width: 700,
      height: 400,
    },
  },
  {
    id: "useless-box",
    title: "Useless Box",
    status: "completed",
    description:
      "A fun electronics project: a box that turns itself off when you turn it on! It has other modes like a shy box.",
    category: "ee-me",
    media: {
      type: "image",
      src: "/images/Useless_box.jpg",
      alt: "Useless Box Project",
      width: 700,
      height: 400,
    },
  },
  {
    id: "cook-drawing",
    title: '"Cook"',
    status: "completed",
    description:
      "The drawing is that of a friend's dog, his name is Coco, but we call him Coook. This was my first time taking on a serious drawing project.",
    category: "drawings",
    media: {
      type: "image",
      src: "/images/Cook.jpg",
      alt: "Cook drawing",
      width: 700,
      height: 400,
    },
  },
  {
    id: "unnamed-drawing",
    title: '"Unnamed"',
    status: "completed",
    description:
      "The drawing was inspired by my research paper about patients suffering from schizophrenia.",
    category: "drawings",
    media: {
      type: "image",
      src: "/images/Ghost.jpg",
      alt: "Unnamed drawing",
      width: 700,
      height: 400,
    },
  },
];
