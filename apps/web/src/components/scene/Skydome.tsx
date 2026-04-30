import { useMemo } from "react";
import { BackSide, Color } from "three";

// Cyberpunk light-pollution sky. Three-stop gradient based on view-Y:
// deep purple-black at zenith, magenta-purple at mid, sodium-orange at
// horizon. Edgerunners / Blade Runner 2049 reference; never starry-blue.
//
// fog: false on the shader so foreground geometry fogs INTO the
// horizon color while the dome itself stays unfogged. Sphere is huge
// (radius 280) and rendered backface so the camera always sits inside
// it; the lower hemisphere is hidden behind the ground plane.

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
uniform vec3 uHorizon;

void main() {
  vec3 dir = normalize(vWorldPos);
  float t = clamp(dir.y, 0.0, 1.0);

  vec3 col;
  if (t < 0.4) {
    col = mix(uHorizon, uMid, smoothstep(0.0, 0.4, t));
  } else {
    col = mix(uMid, uZenith, smoothstep(0.4, 1.0, t));
  }

  gl_FragColor = vec4(col, 1.0);
}
`;

export function Skydome() {
  const uniforms = useMemo(
    () => ({
      uZenith: { value: new Color("#0a0612") },
      uMid: { value: new Color("#3a1040") },
      uHorizon: { value: new Color("#ff5a1f") },
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
