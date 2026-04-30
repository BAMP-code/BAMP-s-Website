import { useLayoutEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Object3D } from "three";
import type { InstancedMesh } from "three";

// Rain particles. Thin elongated quads falling along Y, recycled to
// the top once they pass below the ground. Additive-light blue under
// bloom reads as wet-night neon refraction. Real wet-asphalt reflection
// plane lands with the audio + thunder pass.
const NUM_DROPS = 1200;
const AREA_W = 50;
const AREA_H_TOP = 30;
const AREA_H_BOTTOM = -2;
const AREA_Z_NEAR = 4;
const AREA_Z_FAR = -45;
const FALL_BASE = 14;
const FALL_SPREAD = 9;

const dummy = new Object3D();

type Drop = { x: number; y: number; z: number; speed: number };

function spawnDrop(): Drop {
  return {
    x: (Math.random() - 0.5) * AREA_W,
    y: AREA_H_BOTTOM + Math.random() * (AREA_H_TOP - AREA_H_BOTTOM),
    z: AREA_Z_FAR + Math.random() * (AREA_Z_NEAR - AREA_Z_FAR),
    speed: FALL_BASE + Math.random() * FALL_SPREAD,
  };
}

export function Rain() {
  const ref = useRef<InstancedMesh>(null);
  const drops = useMemo<Drop[]>(
    () => Array.from({ length: NUM_DROPS }, spawnDrop),
    [],
  );

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    drops.forEach((d, i) => {
      dummy.position.set(d.x, d.y, d.z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, [drops]);

  useFrame((_, delta) => {
    const mesh = ref.current;
    if (!mesh) return;
    const dt = Math.min(delta, 0.05);

    for (let i = 0; i < drops.length; i++) {
      const d = drops[i];
      d.y -= d.speed * dt;
      if (d.y < AREA_H_BOTTOM) {
        // Recycle to top with a fresh X to avoid striped patterns.
        d.y = AREA_H_TOP;
        d.x = (Math.random() - 0.5) * AREA_W;
      }
      dummy.position.set(d.x, d.y, d.z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, NUM_DROPS]}>
      <planeGeometry args={[0.018, 0.55]} />
      <meshBasicMaterial
        color="#9ec2ff"
        transparent
        opacity={0.55}
        toneMapped={false}
        depthWrite={false}
      />
    </instancedMesh>
  );
}
