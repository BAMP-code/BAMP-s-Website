import { useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import type { DirectionalLight } from "three";

// Directional light that pulses high-intensity briefly when the audio
// module dispatches "thunder:flash". The pulse is a fast rise (~60 ms)
// followed by an exponential decay (~700 ms) so it reads as a real
// lightning flash rather than a switch.
const PEAK_INTENSITY = 6.5;
const RISE_MS = 60;
const DECAY_MS = 700;

export function Lightning() {
  const lightRef = useRef<DirectionalLight>(null);
  // Wall-clock timestamp when the most recent flash started, or 0.
  const flashStartRef = useRef(0);

  useEffect(() => {
    const onFlash = () => {
      flashStartRef.current = performance.now();
    };
    window.addEventListener("thunder:flash", onFlash);
    return () => window.removeEventListener("thunder:flash", onFlash);
  }, []);

  useFrame(() => {
    const light = lightRef.current;
    if (!light) return;
    const start = flashStartRef.current;
    if (!start) {
      light.intensity = 0;
      return;
    }
    const elapsed = performance.now() - start;
    if (elapsed < RISE_MS) {
      light.intensity = PEAK_INTENSITY * (elapsed / RISE_MS);
    } else if (elapsed < RISE_MS + DECAY_MS) {
      const t = (elapsed - RISE_MS) / DECAY_MS;
      light.intensity = PEAK_INTENSITY * Math.exp(-4 * t);
    } else {
      light.intensity = 0;
      flashStartRef.current = 0;
    }
  });

  return (
    <directionalLight
      ref={lightRef}
      position={[6, 18, 8]}
      intensity={0}
      color="#dfe9ff"
    />
  );
}
