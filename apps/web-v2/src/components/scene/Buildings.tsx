import { useLayoutEffect, useRef } from "react";
import { Object3D } from "three";
import type { InstancedMesh } from "three";

// Placeholder skyscraper grid. Two flanking rows along Z, varying
// heights, no textures. Real building models with project screens
// land in Phase 3.
const NUM_BUILDINGS = 32;
const dummy = new Object3D();

export function Buildings() {
  const ref = useRef<InstancedMesh>(null);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;

    for (let i = 0; i < NUM_BUILDINGS; i++) {
      const side = i % 2 === 0 ? -1 : 1;
      const zIndex = Math.floor(i / 2);
      const h = 1.5 + ((Math.sin(i * 7.13) + 1) / 2) * 4.5;
      const w = 1.6 + ((Math.cos(i * 4.27) + 1) / 2) * 0.6;
      dummy.scale.set(w, h, w);
      dummy.position.set(side * 3.5, h / 2, -zIndex * 3.5 - 5);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, []);

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, NUM_BUILDINGS]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#1a1a26" roughness={0.85} metalness={0.1} />
    </instancedMesh>
  );
}
