import { useEffect, useRef, useState, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import type { Group, Mesh, MeshStandardMaterial } from "three";
import { blipMessages } from "@/content/blip";
import { scrollProgressRef } from "@/lib/scroll";

// Blade-Runner-style ad-blimp. Procedural assembly approximating the
// Spinner anatomy: stretched cylindrical hull with panel-line emissive
// detail, tail fins, hanging gondola pod, twin antenna masts, full
// running-light kit (port red / starboard green / belly strobe / cyan
// edge lights), three searchlights, and a SUSPENDED square LED screen
// on visible cables — not the angled flat panel of the previous pass.
//
// Sized to feel imposing without dominating: ~14 units long, ~15 below
// the camera at scroll=0, drifting overhead as the camera descends.

const MESSAGE_INTERVAL_MS = 3000;
const GLITCH_DURATION_MS = 220;
const HULL_HALF_LENGTH = 6.5;
const HULL_RADIUS = 1.05;
const SIDE_LIGHT_COUNT = 14;
const PANEL_LINE_SHADER_HEADER = /* glsl */ `
varying vec3 vLocalPos;
varying vec3 vLocalNormal;
`;
const PANEL_LINE_SHADER_VERTEX = /* glsl */ `
vLocalPos = position;
vLocalNormal = normal;
`;
const PANEL_LINE_SHADER_FRAGMENT = /* glsl */ `
// Panel grooves: horizontal rings spaced ~0.6 m apart; vertical lines
// every ~30° around the hull. Subtle but reads as "this is hardware,
// not a plastic toy."
float ringSpacing = 0.6;
float ring = step(0.93, abs(sin(vLocalPos.x * 3.14159 / ringSpacing)));
float angle = atan(vLocalPos.z, vLocalPos.y);
float vertical = step(0.97, abs(sin(angle * 6.0)));
float lines = max(ring, vertical);
totalEmissiveRadiance += vec3(0.42, 0.62, 0.85) * lines * 0.4;
diffuseColor.rgb *= mix(1.0, 0.55, lines);
`;

export function Blip() {
  const groupRef = useRef<Group>(null);
  const strobeRef = useRef<Mesh>(null);
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
    const p = scrollProgressRef.current;

    // Cinematic drift — Blip is the OPENING shot. Prominent at scroll
    // 0, recedes up and back as the camera descends so by the time the
    // viewer's halfway through the scene the Blip is a distant hazy
    // silhouette in the upper third. Matches the storyboard in
    // REFACTOR_VISION.md §1: "Blip drifts up + behind."
    const driftT = Math.min(p / 0.35, 1);
    const driftEase = 1 - Math.pow(1 - driftT, 2);
    const baseY = 32 + driftEase * 28; // 32 → 60
    const baseZ = -8 - driftEase * 70; // -8 → -78

    if (groupRef.current) {
      groupRef.current.position.x = Math.sin(t * 0.11) * 0.6;
      groupRef.current.position.y = baseY + Math.sin(t * 0.32) * 0.5;
      groupRef.current.position.z = baseZ;
      groupRef.current.rotation.z = Math.sin(t * 0.18) * 0.04;
      groupRef.current.rotation.y = Math.sin(t * 0.11) * 0.06;
    }
    // Belly strobe: 0.5 Hz hard blink.
    if (strobeRef.current) {
      const mat = strobeRef.current.material as MeshStandardMaterial;
      const on = Math.sin(t * Math.PI) > 0.6;
      mat.emissiveIntensity = on ? 6 : 0.05;
    }
  });

  const message = blipMessages[messageIndex] ?? "";

  // Cyan running lights along the lower hull edge.
  const sideLights = useMemo(
    () =>
      Array.from({ length: SIDE_LIGHT_COUNT }, (_, i) => {
        const xRange = HULL_HALF_LENGTH * 1.85;
        return -xRange / 2 + (xRange * i) / (SIDE_LIGHT_COUNT - 1);
      }),
    [],
  );

  return (
    <group ref={groupRef} position={[0, 32, -8]}>
      {/* Main hull. Stretched cylinder along X; panel lines via
          onBeforeCompile. */}
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry
          args={[HULL_RADIUS, HULL_RADIUS * 0.85, HULL_HALF_LENGTH * 2, 24]}
        />
        <meshStandardMaterial
          color="#1a1a26"
          emissive="#0a0a18"
          emissiveIntensity={0.55}
          roughness={0.42}
          metalness={0.78}
          onBeforeCompile={(shader) => {
            shader.vertexShader = shader.vertexShader.replace(
              "#include <common>",
              `#include <common>\n${PANEL_LINE_SHADER_HEADER}`,
            );
            shader.vertexShader = shader.vertexShader.replace(
              "#include <begin_vertex>",
              `#include <begin_vertex>\n${PANEL_LINE_SHADER_VERTEX}`,
            );
            shader.fragmentShader = shader.fragmentShader.replace(
              "#include <common>",
              `#include <common>\n${PANEL_LINE_SHADER_HEADER}`,
            );
            shader.fragmentShader = shader.fragmentShader.replace(
              "#include <emissivemap_fragment>",
              `#include <emissivemap_fragment>\n${PANEL_LINE_SHADER_FRAGMENT}`,
            );
          }}
        />
      </mesh>

      {/* Tapered nose cone. */}
      <mesh
        rotation={[0, 0, -Math.PI / 2]}
        position={[HULL_HALF_LENGTH + 0.6, 0, 0]}
      >
        <cylinderGeometry args={[0.4, HULL_RADIUS * 0.9, 1.4, 16]} />
        <meshStandardMaterial color="#16161e" roughness={0.45} metalness={0.7} />
      </mesh>

      {/* Rear engine block + glow. */}
      <mesh position={[-HULL_HALF_LENGTH - 0.4, 0, 0]}>
        <boxGeometry args={[0.9, 1.0, 1.4]} />
        <meshStandardMaterial color="#0a0a14" roughness={0.55} metalness={0.55} />
      </mesh>
      <mesh
        position={[-HULL_HALF_LENGTH - 0.92, 0, 0]}
        rotation={[0, Math.PI / 2, 0]}
      >
        <circleGeometry args={[0.36, 24]} />
        <meshBasicMaterial color="#ff5a00" toneMapped={false} />
      </mesh>

      {/* Tail fins — vertical + 2 horizontal, all aft of the engine. */}
      <mesh position={[-HULL_HALF_LENGTH + 0.5, 1.2, 0]}>
        <boxGeometry args={[1.6, 1.4, 0.1]} />
        <meshStandardMaterial color="#16161e" metalness={0.6} roughness={0.5} />
      </mesh>
      <mesh position={[-HULL_HALF_LENGTH + 0.5, -1.05, 0]}>
        <boxGeometry args={[1.6, 0.8, 0.1]} />
        <meshStandardMaterial color="#16161e" metalness={0.6} roughness={0.5} />
      </mesh>
      <mesh position={[-HULL_HALF_LENGTH + 0.5, 0, 1.0]}>
        <boxGeometry args={[1.6, 0.1, 1.2]} />
        <meshStandardMaterial color="#16161e" metalness={0.6} roughness={0.5} />
      </mesh>
      <mesh position={[-HULL_HALF_LENGTH + 0.5, 0, -1.0]}>
        <boxGeometry args={[1.6, 0.1, 1.2]} />
        <meshStandardMaterial color="#16161e" metalness={0.6} roughness={0.5} />
      </mesh>

      {/* Top antenna masts + magenta tip beacons. */}
      <mesh position={[-2.3, 1.25, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 1.9, 8]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      <mesh position={[1.7, 1.3, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 2.1, 8]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      <mesh position={[-2.3, 2.25, 0]}>
        <sphereGeometry args={[0.09, 12, 12]} />
        <meshStandardMaterial
          color="#ff2bd6"
          emissive="#ff2bd6"
          emissiveIntensity={3.0}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[1.7, 2.4, 0]}>
        <sphereGeometry args={[0.09, 12, 12]} />
        <meshStandardMaterial
          color="#ff2bd6"
          emissive="#ff2bd6"
          emissiveIntensity={3.0}
          toneMapped={false}
        />
      </mesh>

      {/* Hanging gondola pod under the front of the hull. */}
      <mesh position={[2.0, -1.65, 0]}>
        <boxGeometry args={[2.0, 0.8, 1.1]} />
        <meshStandardMaterial color="#0a0a14" metalness={0.6} roughness={0.55} />
      </mesh>
      {/* Gondola support struts. */}
      <mesh position={[1.4, -0.95, 0.4]}>
        <cylinderGeometry args={[0.04, 0.04, 0.95, 6]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      <mesh position={[2.6, -0.95, 0.4]}>
        <cylinderGeometry args={[0.04, 0.04, 0.95, 6]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      <mesh position={[1.4, -0.95, -0.4]}>
        <cylinderGeometry args={[0.04, 0.04, 0.95, 6]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      <mesh position={[2.6, -0.95, -0.4]}>
        <cylinderGeometry args={[0.04, 0.04, 0.95, 6]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} />
      </mesh>
      {/* Gondola front window. */}
      <mesh position={[3.02, -1.65, 0]}>
        <planeGeometry args={[0.6, 0.4]} />
        <meshBasicMaterial color="#ffae42" toneMapped={false} />
      </mesh>

      {/* Cyan running lights along both flanks. */}
      {sideLights.map((x, i) => (
        <group key={`light-${i}`}>
          <mesh position={[x, -0.85, 0.92]}>
            <sphereGeometry args={[0.06, 8, 8]} />
            <meshStandardMaterial
              color="#00f6ff"
              emissive="#00f6ff"
              emissiveIntensity={2.6}
              toneMapped={false}
            />
          </mesh>
          <mesh position={[x, -0.85, -0.92]}>
            <sphereGeometry args={[0.06, 8, 8]} />
            <meshStandardMaterial
              color="#00f6ff"
              emissive="#00f6ff"
              emissiveIntensity={2.6}
              toneMapped={false}
            />
          </mesh>
        </group>
      ))}

      {/* Port (red) + starboard (green) navigation lights at the wingtips. */}
      <mesh position={[1.2, -0.4, 1.55]}>
        <sphereGeometry args={[0.13, 12, 12]} />
        <meshStandardMaterial
          color="#00ff5a"
          emissive="#00ff5a"
          emissiveIntensity={3.5}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[1.2, -0.4, -1.55]}>
        <sphereGeometry args={[0.13, 12, 12]} />
        <meshStandardMaterial
          color="#ff2222"
          emissive="#ff2222"
          emissiveIntensity={3.5}
          toneMapped={false}
        />
      </mesh>

      {/* Belly strobe — blinks on at ~0.5 Hz. */}
      <mesh ref={strobeRef} position={[-1.0, -1.05, 0]}>
        <sphereGeometry args={[0.15, 12, 12]} />
        <meshStandardMaterial
          color="#ffffff"
          emissive="#ffffff"
          emissiveIntensity={0.05}
          toneMapped={false}
        />
      </mesh>

      {/* Searchlight cones — three of them, fanning down. */}
      {[-2.2, 0.2, 2.4].map((cx, i) => (
        <mesh
          key={`searchlight-${i}`}
          position={[cx, -3.5, 0]}
          rotation={[Math.PI, 0, 0]}
        >
          <coneGeometry args={[1.5, 5.6, 22, 1, true]} />
          <meshBasicMaterial
            color="#aac8ff"
            transparent
            opacity={0.16}
            side={2}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      ))}

      {/* SUSPENDED LED billboard — square panel hanging below the hull
          on visible cables, like the Blade Runner reference. Sized big
          (7 × 5) because the Blip is now further from the camera and
          the screen is the readable element. */}
      <group position={[-0.5, -5.6, 0]}>
        {/* Suspension cables. */}
        <mesh position={[-3.0, 2.4, 0]}>
          <cylinderGeometry args={[0.025, 0.025, 4.8, 6]} />
          <meshStandardMaterial color="#28283c" metalness={0.6} />
        </mesh>
        <mesh position={[3.0, 2.4, 0]}>
          <cylinderGeometry args={[0.025, 0.025, 4.8, 6]} />
          <meshStandardMaterial color="#28283c" metalness={0.6} />
        </mesh>
        {/* Frame. */}
        <mesh>
          <boxGeometry args={[7.0, 5.0, 0.22]} />
          <meshStandardMaterial
            color="#0e0e18"
            roughness={0.45}
            metalness={0.78}
            emissive="#1a1a26"
            emissiveIntensity={0.35}
          />
        </mesh>
        {/* Screen face. */}
        <mesh position={[0, 0, 0.115]}>
          <planeGeometry args={[6.6, 4.6]} />
          <meshBasicMaterial
            color="#040410"
            opacity={glitching ? 0.45 : 1}
            transparent
            toneMapped={false}
          />
        </mesh>
        {/* Message text. */}
        <Text
          position={[0, 0.95, 0.125]}
          fontSize={1.7}
          color="#00f6ff"
          outlineWidth={0.04}
          outlineColor="#00f6ff"
          outlineOpacity={glitching ? 0.1 : 0.6}
          fillOpacity={glitching ? 0.25 : 1}
          anchorX="center"
          anchorY="middle"
        >
          {message}
        </Text>
        {/* Strapline. */}
        <Text
          position={[0, -1.4, 0.125]}
          fontSize={0.46}
          color="#ff2bd6"
          outlineWidth={0.009}
          outlineColor="#ff2bd6"
          outlineOpacity={0.45}
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
