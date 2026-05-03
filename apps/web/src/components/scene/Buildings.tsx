import { useLayoutEffect, useRef } from "react";
import { Object3D } from "three";
import type { InstancedMesh } from "three";
import { projects } from "@/content/projects";
import type { Project, ProjectCategory } from "@/lib/types";
import { ProjectBuilding } from "./ProjectBuilding";
import { FacadeMaterial } from "./FacadeMaterial";
import { NeonSign } from "./NeonSign";

// Project buildings live on a tiered descent:
//  - Hero (i = 0..2): largest towers, closest to the camera path,
//    framing the start of the descent.
//  - Mid (i = 3..6): standard scale, varied X jitter, the middle of
//    the scroll runway.
//  - Background (i = 7..9): smaller, pushed out wider, integrated with
//    the filler skyline.
//
// Heights are still derived from each project's media aspect ratio
// (so a portrait poster gets a tall narrow tower) but multiplied by a
// tier scale.
const SCREEN_BASE_WIDTH = 7.2;
const BUILDING_BASE_BELOW_SCREEN = 14;
const BUILDING_BASE_ABOVE_SCREEN = 4.5;

// Category accent colors — wayfinding for the descent.
const CATEGORY_COLOR: Record<ProjectCategory, string> = {
  cs: "#00f6ff", // tech cyan
  "ee-me": "#ffae42", // sodium-vapor amber
  drawings: "#ff2bd6", // magenta
};

type Tier = "hero" | "mid" | "background";

function tierFor(i: number): Tier {
  if (i < 3) return "hero";
  if (i < 7) return "mid";
  return "background";
}

const TIER_HEIGHT_SCALE: Record<Tier, number> = {
  hero: 1.45,
  mid: 1.0,
  background: 0.7,
};
const TIER_WIDTH_SCALE: Record<Tier, number> = {
  hero: 1.25,
  mid: 1.0,
  background: 0.85,
};

type Layout = {
  x: number;
  z: number;
  width: number;
  height: number;
  screenWidth: number;
  screenHeight: number;
  screenY: number;
  accent: string;
  tier: Tier;
};

function layoutForProject(i: number, project: Project): Layout {
  const tier = tierFor(i);
  const aspect = project.media.width / project.media.height;

  // Screen size — driven by aspect, capped on portraits, scaled per tier.
  const baseScreenWidth = SCREEN_BASE_WIDTH * TIER_WIDTH_SCALE[tier];
  const baseScreenHeight = Math.min(baseScreenWidth / aspect, 16);

  const heightScale = TIER_HEIGHT_SCALE[tier];
  const buildingHeight =
    (BUILDING_BASE_BELOW_SCREEN + baseScreenHeight + BUILDING_BASE_ABOVE_SCREEN) *
    heightScale;
  const widthJitter = 0.4 + ((Math.sin(i * 5.13) + 1) / 2) * 1.2;
  const buildingWidth = baseScreenWidth + widthJitter;

  // X distance from camera path: heroes hug close, background pushes out.
  const side = i % 2 === 0 ? -1 : 1;
  const xJitter = ((Math.cos(i * 3.7) + 1) / 2) * 0.6;
  let x: number;
  if (tier === "hero") {
    x = side * (3.0 + xJitter * 0.4);
  } else if (tier === "mid") {
    x = side * (3.4 + xJitter);
  } else {
    x = side * (5.6 + xJitter * 1.4);
  }

  // Z spacing — heroes spread further apart for impact.
  const zStep = tier === "hero" ? 6.5 : 4.6;
  const zStart = -10;
  let z = zStart;
  for (let j = 0; j < i; j++) {
    z -= tierFor(j) === "hero" ? 6.5 : 4.6;
  }
  z -= zStep;
  z += ((Math.sin(i * 1.7) + 1) / 2) * 0.9;

  // The screen Y stays anchored to the building's lower portion so
  // taller buildings have more space above the screen — the audit says
  // "building wraps the screen with a body roughly 2× the screen height."
  const screenY = BUILDING_BASE_BELOW_SCREEN + baseScreenHeight / 2;

  return {
    x,
    z,
    width: buildingWidth,
    height: buildingHeight,
    screenWidth: baseScreenWidth,
    screenHeight: baseScreenHeight,
    screenY,
    accent: CATEGORY_COLOR[project.category],
    tier,
  };
}

const NUM_FILLER = 80;
const dummy = new Object3D();

function buildFillerMatrix(i: number) {
  const seed = i * 12.97 + 41.3;
  const r = (Math.sin(seed) + 1) / 2;
  const r2 = (Math.cos(seed * 1.7) + 1) / 2;
  const r3 = (Math.sin(seed * 3.1) + 1) / 2;
  const side = i % 2 === 0 ? -1 : 1;
  const x = side * (10 + r * 14);
  const z = -6 - r2 * 60;
  const w = 2.2 + r3 * 2.4;
  const h = 8 + r * 32;
  dummy.scale.set(w, h, w);
  dummy.position.set(x, h / 2, z);
  dummy.rotation.set(0, r3 * Math.PI, 0);
  dummy.updateMatrix();
  return dummy.matrix;
}

export function Buildings() {
  const fillerRef = useRef<InstancedMesh>(null);

  useLayoutEffect(() => {
    const mesh = fillerRef.current;
    if (!mesh) return;
    for (let i = 0; i < NUM_FILLER; i++) {
      mesh.setMatrixAt(i, buildFillerMatrix(i));
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, []);

  return (
    <group>
      {/* Decorative outer skyline — towers far behind / beside the lane. */}
      <instancedMesh ref={fillerRef} args={[undefined, undefined, NUM_FILLER]}>
        <boxGeometry args={[1, 1, 1]} />
        <FacadeMaterial roughness={0.85} metalness={0.12} />
      </instancedMesh>

      {/* Project lane — tiered descent. */}
      {projects.map((project, i) => {
        const layout = layoutForProject(i, project);
        return (
          <ProjectBuilding
            key={project.id}
            project={project}
            position={[layout.x, 0, layout.z]}
            height={layout.height}
            width={layout.width}
            screenWidth={layout.screenWidth}
            screenHeight={layout.screenHeight}
            screenY={layout.screenY}
            accent={layout.accent}
          />
        );
      })}

      {/* Modeled neon signage on filler / outer skyline buildings.
          Each sign carries a paired off-screen point light so the
          building behind picks up the sign's color — the
          Shibuya/Mongkok cue per the realism research. Sign text is
          intentionally short Asian-megacity style. */}
      <NeonSign
        text="NEON"
        position={[-9, 16, -10]}
        rotation={[0, Math.PI * 0.45, 0]}
        color="#ff2bd6"
        scale={1.2}
      />
      <NeonSign
        text="VOLT"
        position={[10, 12, -18]}
        rotation={[0, -Math.PI * 0.4, 0]}
        color="#00f6ff"
        scale={1.1}
      />
      <NeonSign
        text="2049"
        position={[-12, 22, -22]}
        rotation={[0, Math.PI * 0.5, 0]}
        color="#ffae42"
        scale={1.0}
      />
      <NeonSign
        text="SUSHI"
        position={[11, 18, -32]}
        rotation={[0, -Math.PI * 0.45, 0]}
        color="#ff2222"
        scale={1.1}
      />
      <NeonSign
        text="OPEN"
        position={[-14, 8, -36]}
        rotation={[0, Math.PI * 0.4, 0]}
        color="#00ff88"
        scale={0.9}
      />
      <NeonSign
        text="404"
        position={[13, 26, -45]}
        rotation={[0, -Math.PI * 0.5, 0]}
        color="#ff2bd6"
        scale={1.0}
      />

      {/* Wet-asphalt ground plane. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, -30]}>
        <planeGeometry args={[160, 200]} />
        <meshStandardMaterial
          color="#04040a"
          roughness={0.55}
          metalness={0.35}
        />
      </mesh>
    </group>
  );
}
