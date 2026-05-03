import { Suspense, useMemo, useRef } from "react";
import { useTexture, Text } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import type { ShaderMaterial, Texture } from "three";
import type { Project } from "@/lib/types";
import { screenVertexShader, screenFragmentShader } from "./screen-shader";
import { FacadeMaterial } from "./FacadeMaterial";
import { RooftopGreebles } from "./RooftopGreebles";

type Props = {
  project: Project;
  position: [number, number, number];
  height: number;
  width: number;
  screenWidth: number;
  screenHeight: number;
  screenY: number;
  /** Category accent color (cyan/amber/magenta) used for the title strip. */
  accent: string;
};

// One building = one project. Tall rectangular block; the +z face
// hosts a screen with the project's poster (still image) or, for
// video projects, a "demo loading" placeholder until the video is
// re-encoded (audit task #11).
//
// All buildings face the camera (+z); they're positioned along the
// descent corridor at varying X / Z / heights so the camera passes
// each at a different scroll position. Screen size + Y position are
// computed by Buildings.tsx from the project's media aspect.
export function ProjectBuilding({
  project,
  position,
  height,
  width,
  screenWidth,
  screenHeight,
  screenY,
  accent,
}: Props) {
  const depth = width * 0.85;
  const screenZ = depth / 2 + 0.005;
  // Title strip sits just below the screen.
  const titleY = screenY - screenHeight / 2 - 0.55;

  return (
    <group position={position}>
      {/* Building shell — procedural facade with emissive window grid. */}
      <mesh position={[0, height / 2, 0]}>
        <boxGeometry args={[width, height, depth]} />
        <FacadeMaterial />
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
            width={screenWidth - 0.06}
            height={screenHeight - 0.06}
            position={[0, screenY, screenZ + 0.008]}
          />
        ) : (
          <ProjectScreenFallback
            label={project.title}
            width={screenWidth - 0.06}
            height={screenHeight - 0.06}
            position={[0, screenY, screenZ + 0.008]}
          />
        )}
      </Suspense>

      {/* Project title strip below the screen. Color matches the
          project's category for at-a-glance wayfinding. */}
      <Text
        position={[0, titleY, screenZ + 0.008]}
        fontSize={Math.min(0.42, screenWidth * 0.06)}
        color={accent}
        outlineColor={accent}
        outlineWidth={0.005}
        outlineOpacity={0.55}
        maxWidth={screenWidth}
        textAlign="center"
        anchorX="center"
        anchorY="middle"
      >
        {project.title.toUpperCase()}
      </Text>

      {/* Rooftop machinery — kills the perfectly-flat-roof tell. */}
      <RooftopGreebles
        seed={project.id.length * 17 + project.title.charCodeAt(0)}
        width={width * 0.95}
        depth={depth * 0.95}
        baseY={height}
      />
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
