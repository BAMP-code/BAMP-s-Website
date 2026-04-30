import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { CameraRig } from "./CameraRig";
import { Blip } from "./Blip";
import { Buildings } from "./Buildings";

// Cinematic scroll-descent scene. Mounted via client:only="react" so
// none of this code ships with the initial document. Phase 1
// milestone reached; Phase 2 work in progress (Blip with LED screen +
// neon lighting + bloom for the wet-neon look). Real models / textures
// / atmospherics still ahead.
export function CityScene() {
  return (
    <Canvas
      gl={{
        antialias: true,
        powerPreference: "high-performance",
        // ACES tone mapping is what gives bloom the "filmic" look
        // instead of clipped whites.
        toneMappingExposure: 1.1,
      }}
      dpr={[1, 2]}
      style={{ width: "100%", height: "100%" }}
    >
      <Suspense fallback={null}>
        <fog attach="fog" args={["#070914", 8, 50]} />
        <color attach="background" args={["#070914"]} />

        <PerspectiveCamera makeDefault position={[0, 12, 5]} fov={60} />

        {/* Base lighting — kept low so emissives carry the scene. */}
        <ambientLight intensity={0.12} />
        <directionalLight position={[5, 10, 5]} intensity={0.35} />

        {/* Neon point lights to wash the buildings with cyan + magenta. */}
        <pointLight
          position={[-4, 6, -4]}
          intensity={2.4}
          distance={14}
          color="#00f6ff"
        />
        <pointLight
          position={[4, 4, -10]}
          intensity={2.0}
          distance={14}
          color="#ff2bd6"
        />
        <pointLight
          position={[0, 2, -18]}
          intensity={1.6}
          distance={14}
          color="#ff2bd6"
        />

        <CameraRig />
        <Blip />
        <Buildings />

        <EffectComposer>
          <Bloom
            intensity={0.9}
            luminanceThreshold={0.3}
            luminanceSmoothing={0.4}
            mipmapBlur
          />
        </EffectComposer>
      </Suspense>
    </Canvas>
  );
}
