import { useLayoutEffect, useRef } from "react";
import { Object3D } from "three";
import type { InstancedMesh } from "three";
import { projects } from "@/content/projects";
import type { Project } from "@/lib/types";
import { ProjectBuilding } from "./ProjectBuilding";

// Project buildings are MASSIVE Night-City towers. The screen on
// each is sized from the project's media aspect ratio (so portraits
// like link-app get tall narrow screens, landscapes get wide ones);
// the building wraps the screen with a body roughly 2× the screen
// height. Two flanking lanes at X = ±3 keep them looming close to
// the camera as it descends.
const SCREEN_BASE_WIDTH = 7.2;
const BUILDING_BASE_BELOW_SCREEN = 14;
const BUILDING_BASE_ABOVE_SCREEN = 4.5;

type Layout = {
  x: number;
  z: number;
  width: number;
  height: number;
  screenWidth: number;
  screenHeight: number;
};

function layoutForProject(i: number, project: Project): Layout {
  const aspect = project.media.width / project.media.height;
  // Cap portrait screens so they don't go absurdly tall.
  const screenWidth = SCREEN_BASE_WIDTH;
  const screenHeight = Math.min(screenWidth / aspect, 14);
  const buildingHeight =
    BUILDING_BASE_BELOW_SCREEN + screenHeight + BUILDING_BASE_ABOVE_SCREEN;
  // A bit of width variance so the silhouette isn't uniform.
  const widthJitter = 0.6 + ((Math.sin(i * 5.13) + 1) / 2) * 1.4;
  const buildingWidth = screenWidth + widthJitter;

  // Alternating flanks; small per-building X jitter so the lane
  // doesn't feel like a corridor of identical setbacks.
  const side = i % 2 === 0 ? -1 : 1;
  const xJitter = ((Math.cos(i * 3.7) + 1) / 2) * 0.9;
  const x = side * (3.2 + xJitter);

  // Z spacing: closer near the top of the descent, more spread later.
  const zStep = 4.6;
  const z = -10 - i * zStep + ((Math.sin(i * 1.7) + 1) / 2) * 1.2;

  return {
    x,
    z,
    width: buildingWidth,
    height: buildingHeight,
    screenWidth,
    screenHeight,
  };
}

const NUM_FILLER = 80;
const dummy = new Object3D();

// Decorative filler skyscrapers behind the project lane. Pseudo-random
// layout via deterministic sin/cos so the silhouette is consistent
// across renders. Heights pushed up so the city silhouette towers.
function buildFillerMatrix(i: number) {
  const seed = i * 12.97 + 41.3;
  const r = (Math.sin(seed) + 1) / 2;
  const r2 = (Math.cos(seed * 1.7) + 1) / 2;
  const r3 = (Math.sin(seed * 3.1) + 1) / 2;
  const side = i % 2 === 0 ? -1 : 1;
  // Outer flanks so they don't intrude on the project lane.
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
        <meshStandardMaterial
          color="#0a0a14"
          roughness={0.85}
          metalness={0.12}
        />
      </instancedMesh>

      {/* Project lane — one massive tower per project. */}
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
            screenY={BUILDING_BASE_BELOW_SCREEN + layout.screenHeight / 2}
          />
        );
      })}

      {/* Wet-asphalt ground plane. Extended further so it covers the
          full descent corridor; reflection plane lands later. */}
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
