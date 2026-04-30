import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera } from "@react-three/drei";
import { CameraRig } from "./CameraRig";
import { Blip } from "./Blip";
import { Buildings } from "./Buildings";

// Cinematic scroll-descent scene. Mounted via client:only="react" so
// none of this code ships with the initial document. Phase 1
// milestone: camera Y bound to scroll, placeholder geometry only.
// Real models / textures / atmospherics land in subsequent merges.
export function CityScene() {
  return (
    <Canvas
      gl={{ antialias: true, powerPreference: "high-performance" }}
      dpr={[1, 2]}
      style={{ width: "100%", height: "100%" }}
    >
      <Suspense fallback={null}>
        <fog attach="fog" args={["#070914", 8, 50]} />
        <color attach="background" args={["#070914"]} />
        <PerspectiveCamera makeDefault position={[0, 12, 5]} fov={60} />
        <ambientLight intensity={0.15} />
        <directionalLight position={[5, 10, 5]} intensity={0.6} />
        <CameraRig />
        <Blip />
        <Buildings />
      </Suspense>
    </Canvas>
  );
}
