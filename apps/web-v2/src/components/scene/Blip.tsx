import { useEffect, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import type { Group, Mesh } from "three";
import { blipMessages } from "@/content/blip";

// Ad-balloon with a tethered LED billboard. Real Blip in Phase 2 will
// be modeled (envelope shape, fans, blinking nav lights); for now,
// sphere + flat screen. Screen cycles user-editable messages from
// @/content/blip with a glitch flicker between each.
const MESSAGE_INTERVAL_MS = 3000;
const GLITCH_DURATION_MS = 220;

export function Blip() {
  const groupRef = useRef<Group>(null);
  const balloonRef = useRef<Mesh>(null);
  const [messageIndex, setMessageIndex] = useState(0);
  const [glitching, setGlitching] = useState(false);

  // Cycle messages: glitch flicker → swap message → settle.
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
      // Slow vertical bob + gentle yaw drift.
      groupRef.current.position.y = 11 + Math.sin(t * 0.5) * 0.18;
      groupRef.current.rotation.y = Math.sin(t * 0.15) * 0.12;
    }
    if (balloonRef.current) {
      balloonRef.current.rotation.y = t * 0.05;
    }
  });

  const message = blipMessages[messageIndex] ?? "";

  return (
    <group ref={groupRef} position={[0, 11, -8]}>
      {/* Balloon envelope */}
      <mesh ref={balloonRef} position={[0, 0.9, 0]}>
        <sphereGeometry args={[0.85, 32, 32]} />
        <meshStandardMaterial
          color="#1a1a26"
          emissive="#00f6ff"
          emissiveIntensity={0.18}
          roughness={0.55}
          metalness={0.4}
        />
      </mesh>

      {/* Tether */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.02, 0.02, 0.4, 8]} />
        <meshStandardMaterial color="#28283c" />
      </mesh>

      {/* LED billboard frame */}
      <mesh position={[0, -0.55, 0]}>
        <boxGeometry args={[3.4, 0.95, 0.12]} />
        <meshStandardMaterial
          color="#0a0a14"
          roughness={0.7}
          metalness={0.5}
        />
      </mesh>

      {/* LED screen face (slightly in front of the frame) */}
      <mesh position={[0, -0.55, 0.07]}>
        <planeGeometry args={[3.2, 0.78]} />
        <meshBasicMaterial
          color="#040410"
          opacity={glitching ? 0.4 : 1}
          transparent
        />
      </mesh>

      {/* Message text */}
      <Text
        position={[0, -0.55, 0.08]}
        fontSize={0.42}
        color="#00f6ff"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.012}
        outlineColor="#00f6ff"
        outlineOpacity={glitching ? 0.1 : 0.55}
        fillOpacity={glitching ? 0.25 : 1}
      >
        {message}
      </Text>
    </group>
  );
}
