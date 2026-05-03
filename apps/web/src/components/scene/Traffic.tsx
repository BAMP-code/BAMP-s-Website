import { useLayoutEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Object3D, Color } from "three";
import type { InstancedMesh } from "three";

// Flying-car traffic. Small bright emissive pinpoints traveling along
// horizontal lanes at multiple altitudes. Reds head one way, whites
// the other — the long-exposure light-trail effect everyone knows
// from cyberpunk cinematography (Blade Runner 2049 freeway shots,
// Edgerunners cityscape transitions).
//
// Two instanced meshes (red + white) keep total draw calls at 2 even
// with 80+ moving lights.

type Lane = {
  y: number; // altitude
  xMin: number;
  xMax: number;
  z: number;
  speed: number; // units per second; sign = direction
  count: number;
};

const RED_LANES: Lane[] = [
  { y: 5, xMin: -28, xMax: 28, z: -18, speed: 22, count: 6 },
  { y: 13, xMin: -28, xMax: 28, z: -10, speed: 26, count: 5 },
  { y: 22, xMin: -28, xMax: 28, z: -28, speed: 28, count: 5 },
  { y: 30, xMin: -28, xMax: 28, z: -42, speed: 32, count: 4 },
];

const WHITE_LANES: Lane[] = [
  { y: 7, xMin: -28, xMax: 28, z: -8, speed: -22, count: 6 },
  { y: 16, xMin: -28, xMax: 28, z: -22, speed: -26, count: 5 },
  { y: 26, xMin: -28, xMax: 28, z: -36, speed: -28, count: 5 },
  { y: 34, xMin: -28, xMax: 28, z: -52, speed: -32, count: 4 },
];

const dummy = new Object3D();

type StreakState = {
  x: number;
  y: number;
  z: number;
  speed: number;
  xMin: number;
  xMax: number;
};

function buildStates(lanes: Lane[]): StreakState[] {
  const out: StreakState[] = [];
  for (const lane of lanes) {
    for (let i = 0; i < lane.count; i++) {
      // Spread initial X across the lane so they don't all bunch.
      const t = (i + Math.random() * 0.5) / lane.count;
      const x = lane.xMin + (lane.xMax - lane.xMin) * t;
      out.push({
        x,
        y: lane.y + (Math.random() - 0.5) * 0.6,
        z: lane.z + (Math.random() - 0.5) * 1.5,
        speed: lane.speed,
        xMin: lane.xMin,
        xMax: lane.xMax,
      });
    }
  }
  return out;
}

function StreakPair({
  states,
  color,
  scaleY = 0.06,
  scaleZ = 0.06,
  scaleX = 1.2,
  emissiveIntensity = 6,
}: {
  states: StreakState[];
  color: string;
  scaleY?: number;
  scaleZ?: number;
  scaleX?: number;
  emissiveIntensity?: number;
}) {
  const meshRef = useRef<InstancedMesh>(null);
  const colorObj = useMemo(() => new Color(color), [color]);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    states.forEach((s, i) => {
      dummy.position.set(s.x, s.y, s.z);
      dummy.scale.set(scaleX, scaleY, scaleZ);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, [states, scaleX, scaleY, scaleZ]);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dt = Math.min(delta, 0.05);
    for (let i = 0; i < states.length; i++) {
      const s = states[i];
      s.x += s.speed * dt;
      // Wrap when the streak passes the lane end.
      if (s.speed > 0 && s.x > s.xMax) s.x = s.xMin;
      else if (s.speed < 0 && s.x < s.xMin) s.x = s.xMax;
      dummy.position.set(s.x, s.y, s.z);
      dummy.scale.set(scaleX, scaleY, scaleZ);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, states.length]}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        color={colorObj}
        emissive={colorObj}
        emissiveIntensity={emissiveIntensity}
        toneMapped={false}
      />
    </instancedMesh>
  );
}

export function Traffic() {
  const redStates = useMemo(() => buildStates(RED_LANES), []);
  const whiteStates = useMemo(() => buildStates(WHITE_LANES), []);

  return (
    <group>
      <StreakPair states={redStates} color="#ff3030" />
      <StreakPair states={whiteStates} color="#ffffff" />
    </group>
  );
}
