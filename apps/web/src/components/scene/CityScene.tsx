import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera, Environment } from "@react-three/drei";
import {
  EffectComposer,
  Bloom,
  ChromaticAberration,
  Noise,
  Vignette,
  DepthOfField,
} from "@react-three/postprocessing";
import { KernelSize, BlendFunction } from "postprocessing";
import { Vector2 } from "three";
import { CameraRig } from "./CameraRig";
import { Blip } from "./Blip";
import { Buildings } from "./Buildings";
import { Rain } from "./Rain";
import { Lightning } from "./Lightning";
import { Skydome } from "./Skydome";

// Cinematic scroll-descent scene. Full realism stack now:
//   - Image-based lighting from a Poly Haven night HDRI (satara_night
//     1k, 1.8 MB). Provides the directionless ambient color tint that
//     makes PBR materials stop reading as plastic.
//   - Hemisphere + cool-moonlight directional replace the point-light
//     spam from earlier passes; the HDRI is doing most of the ambient
//     work now.
//   - Tone mapping exposure dropped to 0.65 — night scenes need to be
//     *dark* with bright neon punching through.
//   - Selective tight-threshold bloom (0.92) so only true emissives
//     glow; chromatic aberration + film-grain Noise + vignette sell
//     "this is a photographed frame, not a 3D render."
export function CityScene() {
  return (
    <Canvas
      gl={{
        antialias: true,
        powerPreference: "high-performance",
        toneMappingExposure: 0.65,
      }}
      dpr={[1, 2]}
      style={{ width: "100%", height: "100%" }}
    >
      <Suspense fallback={null}>
        {/* Volumetric fog + sky. Fog matches the warm-dim horizon glow. */}
        <fogExp2 attach="fog" args={["#0a0a12", 0.018]} />
        <Skydome />

        {/* Image-based lighting from a real night HDRI. background=false
            so we keep the procedural Skydome as the visible sky. */}
        <Environment
          files="/hdri/satara_night_1k.hdr"
          resolution={256}
          background={false}
        />

        <PerspectiveCamera
          makeDefault
          position={[0, 32, 5]}
          fov={70}
          near={0.1}
          far={300}
        />

        {/* Lighting rig — minimal now that the HDRI handles ambient.
            One soft hemisphere, one cool moonlight key, and only a
            couple of neon accent point lights instead of the previous
            four. */}
        <hemisphereLight
          color="#0a0c14"
          groundColor="#1a0f08"
          intensity={0.18}
        />
        <directionalLight
          position={[-30, 80, 20]}
          color="#7088aa"
          intensity={0.45}
        />
        <pointLight
          position={[-5, 18, -14]}
          intensity={2.4}
          distance={28}
          color="#00f6ff"
        />
        <pointLight
          position={[5, 8, -32]}
          intensity={2.2}
          distance={28}
          color="#ff2bd6"
        />

        <CameraRig />
        <Lightning />
        <Blip />
        <Buildings />
        <Rain />

        {/* Cinematic post stack — the realism research's #1 lever.
            Tight bloom threshold so only emissive neon glows; subtle
            chromatic aberration + film grain + vignette read as
            "photographic frame" instead of "3D render." */}
        <EffectComposer multisampling={0}>
          {/* DOF: focal plane around the project lane (~15–20 units
              away from camera). Background and very-foreground go soft.
              Subtle bokehScale; we don't want anime-style blur, just
              cinematic falloff. */}
          <DepthOfField
            focusDistance={0.06}
            focalLength={0.05}
            bokehScale={1.6}
            height={480}
          />
          <Bloom
            intensity={0.6}
            luminanceThreshold={0.92}
            luminanceSmoothing={0.025}
            mipmapBlur
            kernelSize={KernelSize.LARGE}
          />
          <ChromaticAberration
            offset={new Vector2(0.0006, 0.0006)}
            radialModulation={false}
            modulationOffset={0.0}
          />
          <Noise
            opacity={0.045}
            premultiply
            blendFunction={BlendFunction.SOFT_LIGHT}
          />
          <Vignette
            eskil={false}
            offset={0.18}
            darkness={0.85}
            blendFunction={BlendFunction.NORMAL}
          />
        </EffectComposer>
      </Suspense>
    </Canvas>
  );
}
