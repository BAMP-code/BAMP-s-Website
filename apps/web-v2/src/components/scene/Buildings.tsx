import { useLayoutEffect, useRef } from "react";
import { Object3D } from "three";
import type { InstancedMesh } from "three";
import { projects } from "@/content/projects";
import { ProjectBuilding } from "./ProjectBuilding";

// Project building positions. Heights chosen so each screen sits at a
// distinct Y altitude, giving the descending camera something to pass
// at every scroll segment. Z depth steadily recedes into the fog.
const projectLayout: { x: number; z: number; height: number; width?: number }[] =
  [
    { x: -4.0, z: -12, height: 14 },
    { x: 4.2, z: -14, height: 12 },
    { x: -4.5, z: -16, height: 10 },
    { x: 4.0, z: -18, height: 9 },
    { x: -4.0, z: -20, height: 8 },
    { x: 4.5, z: -22, height: 7 },
    { x: -4.2, z: -24, height: 6 },
    { x: 4.0, z: -26, height: 5 },
    { x: -4.0, z: -28, height: 4 },
    { x: 4.0, z: -30, height: 3 },
  ];

const NUM_FILLER = 56;
const dummy = new Object3D();

// Decorative filler skyscrapers behind the project lane. Pseudo-random
// layout via deterministic sin/cos so the silhouette is consistent
// across renders.
function buildFillerMatrix(i: number) {
  const seed = i * 12.97 + 41.3;
  const r = (Math.sin(seed) + 1) / 2;
  const r2 = (Math.cos(seed * 1.7) + 1) / 2;
  const r3 = (Math.sin(seed * 3.1) + 1) / 2;
  // Place outside the project flanks (|x| > 7) and across the descent z.
  const side = i % 2 === 0 ? -1 : 1;
  const x = side * (7.5 + r * 9);
  const z = -8 - r2 * 28;
  const w = 1.4 + r3 * 1.4;
  const h = 2 + r * 9;
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
      {/* Decorative outer skyline. */}
      <instancedMesh
        ref={fillerRef}
        args={[undefined, undefined, NUM_FILLER]}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          color="#0a0a14"
          roughness={0.85}
          metalness={0.12}
        />
      </instancedMesh>

      {/* Project lane — one building per project. */}
      {projects.map((project, i) => {
        const layout = projectLayout[i];
        if (!layout) return null;
        return (
          <ProjectBuilding
            key={project.id}
            project={project}
            position={[layout.x, 0, layout.z]}
            height={layout.height}
            width={layout.width}
          />
        );
      })}

      {/* Ground plane — wet asphalt placeholder. Real reflection comes
          with the rain pass. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, -15]}>
        <planeGeometry args={[80, 80]} />
        <meshStandardMaterial
          color="#04040a"
          roughness={0.55}
          metalness={0.35}
        />
      </mesh>
    </group>
  );
}
