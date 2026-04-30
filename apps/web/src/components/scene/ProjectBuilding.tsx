import { Suspense, useMemo, useRef } from "react";
import { useTexture, Text } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import type { ShaderMaterial, Texture } from "three";
import type { Project } from "@/lib/types";
import { screenVertexShader, screenFragmentShader } from "./screen-shader";

type Props = {
  project: Project;
  position: [number, number, number];
  height: number;
  width?: number;
};

const SCREEN_BORDER = 0.16;

// One building = one project. Tall rectangular block; the +z face
// hosts a screen with the project's poster (still image) or, for
// video projects, a "demo loading" placeholder until the video is
// re-encoded (audit task #11).
//
// All buildings face the camera (+z); they're positioned along the
// descent corridor at varying X / Z / heights so the camera passes
// each at a different scroll position.
export function ProjectBuilding({
  project,
  position,
  height,
  width = 2.4,
}: Props) {
  const depth = width * 0.85;
  const screenWidth = width - SCREEN_BORDER * 2;
  const screenHeight = height * 0.42;
  const screenY = height * 0.62;
  const screenZ = depth / 2 + 0.005;

  return (
    <group position={position}>
      {/* Building shell. */}
      <mesh position={[0, height / 2, 0]}>
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial
          color="#0a0a14"
          roughness={0.78}
          metalness={0.32}
          emissive="#070914"
          emissiveIntensity={0.15}
        />
      </mesh>

      {/* Screen frame — slightly recessed dark panel. */}
      <mesh position={[0, screenY, screenZ]}>
        <planeGeometry args={[screenWidth, screenHeight]} />
        <meshStandardMaterial
          color="#020208"
          emissive="#020208"
          emissiveIntensity={0.4}
          roughness={0.6}
          metalness={0.4}
        />
      </mesh>

      {/* Screen content. */}
      <Suspense fallback={null}>
        {project.media.type === "image" ? (
          <ProjectScreenImage
            src={project.media.src}
            width={screenWidth - 0.04}
            height={screenHeight - 0.04}
            position={[0, screenY, screenZ + 0.008]}
          />
        ) : (
          <ProjectScreenFallback
            label={project.title}
            width={screenWidth - 0.04}
            height={screenHeight - 0.04}
            position={[0, screenY, screenZ + 0.008]}
          />
        )}
      </Suspense>

      {/* Project title strip below the screen. */}
      <Text
        position={[0, height * 0.34, screenZ + 0.008]}
        fontSize={0.14}
        color="#00f6ff"
        outlineColor="#00f6ff"
        outlineWidth={0.004}
        outlineOpacity={0.45}
        maxWidth={screenWidth}
        textAlign="center"
        anchorX="center"
        anchorY="middle"
      >
        {project.title.toUpperCase()}
      </Text>
    </group>
  );
}

function ProjectScreenImage({
  src,
  width,
  height,
  position,
}: {
  src: string;
  width: number;
  height: number;
  position: [number, number, number];
}) {
  const texture = useTexture(src) as Texture;
  texture.anisotropy = 8;
  return (
    <ScreenSurface texture={texture} width={width} height={height} position={position} />
  );
}

function ScreenSurface({
  texture,
  width,
  height,
  position,
}: {
  texture: Texture;
  width: number;
  height: number;
  position: [number, number, number];
}) {
  const matRef = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(
    () => ({
      uMap: { value: texture },
      uTime: { value: 0 },
    }),
    [texture],
  );

  useFrame((state) => {
    if (matRef.current) {
      matRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });

  return (
    <mesh position={position}>
      <planeGeometry args={[width, height]} />
      <shaderMaterial
        ref={matRef}
        uniforms={uniforms}
        vertexShader={screenVertexShader}
        fragmentShader={screenFragmentShader}
        toneMapped={false}
      />
    </mesh>
  );
}

function ProjectScreenFallback({
  label,
  width,
  height,
  position,
}: {
  label: string;
  width: number;
  height: number;
  position: [number, number, number];
}) {
  return (
    <group position={position}>
      <mesh>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial color="#070914" toneMapped={false} />
      </mesh>
      <Text
        position={[0, 0, 0.005]}
        fontSize={Math.min(width, height) * 0.16}
        color="#ff2bd6"
        outlineColor="#ff2bd6"
        outlineWidth={0.006}
        outlineOpacity={0.5}
        maxWidth={width * 0.9}
        textAlign="center"
        anchorX="center"
        anchorY="middle"
      >
        {label.toUpperCase()}
      </Text>
      <Text
        position={[0, -height * 0.32, 0.005]}
        fontSize={Math.min(width, height) * 0.07}
        color="#9ba4c7"
        anchorX="center"
        anchorY="middle"
      >
        DEMO LOADING…
      </Text>
    </group>
  );
}
