import { useEffect, useRef, useState, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import type { Group } from "three";
import { blipMessages } from "@/content/blip";

// Blade-Runner-style ad-blimp. Segmented hull along X, antenna masts,
// running lights, two searchlight cones beaming down, and a large LED
// screen mounted beside the hull cycling Bryan's messages
// (`:p`, `hello`, `<3`) with a glitch flicker between each.
const MESSAGE_INTERVAL_MS = 3000;
const GLITCH_DURATION_MS = 220;
const SIDE_LIGHT_COUNT = 10;

export function Blip() {
  const groupRef = useRef<Group>(null);
  const [messageIndex, setMessageIndex] = useState(0);
  const [glitching, setGlitching] = useState(false);

  useEffect(() => {
    if (blipMessages.length <= 1) return;
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      setGlitching(true);
      window.setTimeout(() => {
        if (cancelled) return;
        setMessageIndex((i) => (i + 1) % blipMessages.length);
        setGlitching(false);
      }, GLITCH_DURATION_MS);
    };
    const id = window.setInterval(tick, MESSAGE_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (groupRef.current) {
      // Slow vertical drift + a small roll so the searchlights don't
      // look mechanically static.
      groupRef.current.position.y = 22 + Math.sin(t * 0.32) * 0.35;
      groupRef.current.rotation.z = Math.sin(t * 0.18) * 0.04;
      groupRef.current.rotation.y = Math.sin(t * 0.11) * 0.06;
    }
  });

  const message = blipMessages[messageIndex] ?? "";

  // Running lights along the lower edge of the hull.
  const sideLights = useMemo(
    () =>
      Array.from({ length: SIDE_LIGHT_COUNT }, (_, i) => {
        const xRange = 5.6;
        const x = -xRange / 2 + (xRange * i) / (SIDE_LIGHT_COUNT - 1);
        return x;
      }),
    [],
  );

  return (
    <group ref={groupRef} position={[0, 22, -3]}>
      {/* Main hull — stretched cylinder along X with subtle taper. */}
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.85, 0.7, 6.2, 16]} />
        <meshStandardMaterial
          color="#16161e"
          emissive="#0a0a18"
          emissiveIntensity={0.6}
          roughness={0.45}
          metalness={0.7}
        />
      </mesh>

      {/* Front tapered nose. */}
      <mesh rotation={[0, 0, -Math.PI / 2]} position={[3.6, 0, 0]}>
        <cylinderGeometry args={[0.4, 0.85, 1.2, 16]} />
        <meshStandardMaterial
          color="#16161e"
          roughness={0.45}
          metalness={0.7}
        />
      </mesh>

      {/* Rear engine block. */}
      <mesh position={[-3.4, 0, 0]}>
        <boxGeometry args={[0.6, 0.9, 1.1]} />
        <meshStandardMaterial
          color="#0a0a14"
          roughness={0.55}
          metalness={0.55}
        />
      </mesh>

      {/* Engine glow disc. */}
      <mesh position={[-3.78, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
        <circleGeometry args={[0.32, 24]} />
        <meshBasicMaterial color="#ff5a00" toneMapped={false} />
      </mesh>

      {/* Top antenna masts. */}
      <mesh position={[-2.1, 1.05, 0]}>
        <cylinderGeometry args={[0.045, 0.045, 1.7, 8]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      <mesh position={[1.6, 1.1, 0]}>
        <cylinderGeometry args={[0.045, 0.045, 1.9, 8]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      <mesh position={[-2.1, 1.95, 0]}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshStandardMaterial
          color="#ff2bd6"
          emissive="#ff2bd6"
          emissiveIntensity={2.4}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[1.6, 2.1, 0]}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshStandardMaterial
          color="#ff2bd6"
          emissive="#ff2bd6"
          emissiveIntensity={2.4}
          toneMapped={false}
        />
      </mesh>

      {/* Bottom fin / keel. */}
      <mesh position={[0.4, -0.85, 0]}>
        <boxGeometry args={[3.2, 0.45, 0.08]} />
        <meshStandardMaterial
          color="#0a0a14"
          roughness={0.5}
          metalness={0.7}
        />
      </mesh>

      {/* Running lights along both sides of the hull. */}
      {sideLights.map((x, i) => (
        <group key={`light-${i}`}>
          <mesh position={[x, -0.65, 0.7]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshStandardMaterial
              color="#00f6ff"
              emissive="#00f6ff"
              emissiveIntensity={2.2}
              toneMapped={false}
            />
          </mesh>
          <mesh position={[x, -0.65, -0.7]}>
            <sphereGeometry args={[0.05, 8, 8]} />
            <meshStandardMaterial
              color="#00f6ff"
              emissive="#00f6ff"
              emissiveIntensity={2.2}
              toneMapped={false}
            />
          </mesh>
        </group>
      ))}

      {/* Searchlight cones pointing down. Translucent volumes that
          read as light shafts under bloom. */}
      <mesh position={[-1.6, -3.2, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[1.5, 5.4, 22, 1, true]} />
        <meshBasicMaterial
          color="#aac8ff"
          transparent
          opacity={0.16}
          side={2}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[1.7, -3.2, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[1.5, 5.4, 22, 1, true]} />
        <meshBasicMaterial
          color="#aac8ff"
          transparent
          opacity={0.16}
          side={2}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>

      {/* LED billboard, mounted on the right side, angled toward the
          camera path. Frame + screen + message text + strapline. */}
      <group position={[3.3, 0.05, 1.2]} rotation={[0, -Math.PI * 0.18, 0]}>
        <mesh>
          <boxGeometry args={[3.7, 2.5, 0.16]} />
          <meshStandardMaterial
            color="#0e0e18"
            roughness={0.45}
            metalness={0.75}
            emissive="#1a1a26"
            emissiveIntensity={0.4}
          />
        </mesh>
        <mesh position={[0, 0, 0.085]}>
          <planeGeometry args={[3.45, 2.25]} />
          <meshBasicMaterial
            color="#040410"
            opacity={glitching ? 0.45 : 1}
            transparent
            toneMapped={false}
          />
        </mesh>
        <Text
          position={[0, 0.4, 0.095]}
          fontSize={0.78}
          color="#00f6ff"
          outlineWidth={0.018}
          outlineColor="#00f6ff"
          outlineOpacity={glitching ? 0.1 : 0.6}
          fillOpacity={glitching ? 0.25 : 1}
          anchorX="center"
          anchorY="middle"
        >
          {message}
        </Text>
        <Text
          position={[0, -0.7, 0.095]}
          fontSize={0.22}
          color="#ff2bd6"
          outlineWidth={0.005}
          outlineColor="#ff2bd6"
          outlineOpacity={0.4}
          letterSpacing={0.18}
          anchorX="center"
          anchorY="middle"
        >
          BAMP.CODES
        </Text>
      </group>
    </group>
  );
}
