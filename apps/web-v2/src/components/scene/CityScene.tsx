import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera } from "@react-three/drei";

// Empty scene baseline. Camera + lights only — geometry and the
// scroll-driven camera rig land in the next merge. Mounted as a
// React island via `client:only="react"` so none of this code ships
// with the initial document.
export function CityScene() {
  return (
    <Canvas
      gl={{ antialias: true, powerPreference: "high-performance" }}
      dpr={[1, 2]}
      style={{ width: "100%", height: "100%" }}
    >
      <Suspense fallback={null}>
        <PerspectiveCamera makeDefault position={[0, 0, 5]} fov={60} />
        <ambientLight intensity={0.2} />
        <directionalLight position={[5, 10, 5]} intensity={1.0} />
      </Suspense>
    </Canvas>
  );
}
