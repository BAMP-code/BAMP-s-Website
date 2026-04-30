import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import type { Mesh } from "three";

// Placeholder ad-balloon. Real Blip with LED screen + glitching messages
// lands in Phase 2. For now: cyan emissive sphere that bobs slowly.
export function Blip() {
  const ref = useRef<Mesh>(null);

  useFrame((state) => {
    if (!ref.current) return;
    ref.current.position.y =
      11 + Math.sin(state.clock.elapsedTime * 0.5) * 0.15;
    ref.current.rotation.y = state.clock.elapsedTime * 0.05;
  });

  return (
    <mesh ref={ref} position={[0, 11, -8]}>
      <sphereGeometry args={[0.8, 32, 32]} />
      <meshStandardMaterial
        color="#00f6ff"
        emissive="#00f6ff"
        emissiveIntensity={1.2}
        roughness={0.3}
        metalness={0.6}
      />
    </mesh>
  );
}
