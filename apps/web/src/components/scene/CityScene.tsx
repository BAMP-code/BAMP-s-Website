import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { KernelSize } from "postprocessing";
import { CameraRig } from "./CameraRig";
import { Blip } from "./Blip";
import { Buildings } from "./Buildings";
import { Rain } from "./Rain";
import { Lightning } from "./Lightning";
import { Skydome } from "./Skydome";

// Cinematic scroll-descent scene. Mounted via client:only="react".
// Camera starts high above the project lane (Y=32, looking down) and
// descends to street level (Y=2.2, looking forward). Buildings around
// the corridor are 28–48 units tall.
//
// Sky + fog are tuned to Edgerunners / Blade Runner 2049 light-pollution
// palette: warm purple-orange, never blue. Bloom is configured to only
// catch true emissives (threshold 1.0) — without that, midtones bloom
// and the scene reads "washed out" instead of "neon."
export function CityScene() {
  return (
    <Canvas
      gl={{
        antialias: true,
        powerPreference: "high-performance",
        toneMappingExposure: 1.0,
      }}
      dpr={[1, 2]}
      style={{ width: "100%", height: "100%" }}
    >
      <Suspense fallback={null}>
        {/* FogExp2 with warm-purple color matches Mie scattering and
            the horizon stop of the skydome. Foreground fades INTO the
            sky color rather than into a black void. */}
        <fogExp2 attach="fog" args={["#2a0e3a", 0.014]} />

        <Skydome />

        <PerspectiveCamera
          makeDefault
          position={[0, 32, 5]}
          fov={70}
          near={0.1}
          far={300}
        />

        {/* Base lighting — kept low so emissives carry the scene. */}
        <ambientLight intensity={0.18} />
        <directionalLight position={[5, 30, 5]} intensity={0.4} />

        {/* Neon point lights washing the project lane. These illuminate
            (don't themselves bloom — they're not emissive geometry). */}
        <pointLight
          position={[-5, 22, -10]}
          intensity={4.2}
          distance={30}
          color="#00f6ff"
        />
        <pointLight
          position={[5, 14, -22]}
          intensity={3.8}
          distance={30}
          color="#ff2bd6"
        />
        <pointLight
          position={[-4, 6, -36]}
          intensity={3.4}
          distance={30}
          color="#ff2bd6"
        />
        <pointLight
          position={[4, 4, -50]}
          intensity={3.0}
          distance={30}
          color="#00f6ff"
        />

        <CameraRig />
        <Lightning />
        <Blip />
        <Buildings />
        <Rain />

        {/* Bloom: threshold 1.0 means only pixels brighter than 1.0
            (i.e., true emissives with intensity > 1) glow. Midtones
            stay clean — that's why the scene now reads "neon city" not
            "washed-out fog." Large kernel for soft, wide haloes. */}
        <EffectComposer>
          <Bloom
            intensity={1.4}
            luminanceThreshold={1.0}
            luminanceSmoothing={0.05}
            mipmapBlur
            kernelSize={KernelSize.LARGE}
          />
        </EffectComposer>
      </Suspense>
    </Canvas>
  );
}
