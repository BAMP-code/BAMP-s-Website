import { useMemo } from "react";
import { BackSide, Color } from "three";

// Real-night-city sky. Earlier passes used a tall purple/orange
// gradient — that's the *Edgerunners* re-grade, not what big cities
// actually look like at night. Photo reference (Liam Wong / Tokyo,
// Roger Deakins / Blade Runner 2049, raw HK/Shanghai/NYC observation
// deck shots) shows:
//
//   - Zenith nearly black with a cool cast (#05070d)
//   - Mid-sky a slightly warmer dark gray-brown (#0e0d12)
//   - Horizon a *thin* warm band (#2a1f12 → #4a3520) only 10–15° tall
//
// Stars killed by light pollution — none rendered.
//
// fog: false on the shader so foreground geometry fogs INTO the
// horizon color while the dome itself stays unfogged.

const vertexShader = /* glsl */ `
varying vec3 vWorldPos;
void main() {
  vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = /* glsl */ `
varying vec3 vWorldPos;
uniform vec3 uZenith;
uniform vec3 uMid;
uniform vec3 uHorizonHi;
uniform vec3 uHorizonLo;

void main() {
  vec3 dir = normalize(vWorldPos);
  float y = dir.y;

  vec3 col;
  if (y < 0.08) {
    // Tight horizon glow band (0 → 0.08 view-Y, ~10° elevation).
    // Hotter at the bottom edge, fading up into mid-sky color.
    float t = clamp(y / 0.08, 0.0, 1.0);
    col = mix(uHorizonHi, uMid, smoothstep(0.0, 1.0, t));
    // Even hotter glow right at the horizon line.
    if (y < 0.04) {
      col = mix(uHorizonLo, col, smoothstep(0.0, 0.04, y));
    }
  } else {
    // Smooth dark transition: mid → zenith.
    float t = (y - 0.08) / 0.92;
    col = mix(uMid, uZenith, smoothstep(0.0, 1.0, t));
  }

  gl_FragColor = vec4(col, 1.0);
}
`;

export function Skydome() {
  const uniforms = useMemo(
    () => ({
      uZenith: { value: new Color("#05070d") },
      uMid: { value: new Color("#0e0d12") },
      uHorizonHi: { value: new Color("#2a1f12") },
      uHorizonLo: { value: new Color("#4a3520") },
    }),
    [],
  );

  return (
    <mesh>
      <sphereGeometry args={[280, 32, 16]} />
      <shaderMaterial
        side={BackSide}
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        fog={false}
        depthWrite={false}
      />
    </mesh>
  );
}
