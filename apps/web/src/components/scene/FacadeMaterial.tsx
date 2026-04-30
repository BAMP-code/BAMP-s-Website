import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import type { MeshStandardMaterial } from "three";

// Procedural emissive window-grid material. Used for both project
// building shells and filler skyscrapers.
//
// The shader is injected into MeshStandardMaterial via onBeforeCompile
// so PBR (point-light wash, fog, shadows) still works — only the
// emissive output is augmented with the window grid.
//
// Per-cell hashing drives on/off (~55% lit), warm/cool color mix
// (sodium-amber ↔ tech-cyan), and a slow per-window flicker. Triplanar
// projection keeps window aspect square on each face. Top/bottom faces
// are skipped (the window code looks down the wrong axis up there).
//
// World-space coords mean each building / each instance gets a unique
// pattern automatically — no per-instance attributes needed.

const VERTEX_HEADER = /* glsl */ `
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
`;

const VERTEX_BODY = /* glsl */ `
vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
vWorldNormal = normalize(mat3(modelMatrix) * normal);
`;

const FRAGMENT_HEADER = /* glsl */ `
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
uniform float uFacadeTime;

float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
`;

const FRAGMENT_BODY = /* glsl */ `
// Pick the right pair of axes based on which face we're on.
vec3 absN = abs(vWorldNormal);
vec2 gridUV;
if (absN.x > absN.z && absN.x > absN.y) {
  gridUV = vec2(vWorldPos.z, vWorldPos.y);
} else if (absN.z > absN.y) {
  gridUV = vec2(vWorldPos.x, vWorldPos.y);
} else {
  gridUV = vec2(vWorldPos.x, vWorldPos.z);
}

// Cell size in world units → roughly one window per 0.55 m wide and
// 0.85 m tall. Windows are square-ish but slightly tall.
vec2 cellSize = vec2(0.55, 0.9);
vec2 cell = floor(gridUV / cellSize);
vec2 cellLocal = fract(gridUV / cellSize);

// Inset (border around each window pane).
vec2 winInset = vec2(0.18, 0.13);
float window =
  step(winInset.x, cellLocal.x) * step(cellLocal.x, 1.0 - winInset.x) *
  step(winInset.y, cellLocal.y) * step(cellLocal.y, 1.0 - winInset.y);

// Per-window randomness.
float h1 = hash21(cell);
float h2 = hash21(cell + vec2(7.1, 13.7));
float h3 = hash21(cell + vec2(31.5, 57.3));
float lit = step(0.55, h1);
float flicker = 0.78 + 0.22 * sin(uFacadeTime * (1.5 + h2 * 3.0) + h1 * 6.28);
// A few windows do a faster harsh flicker (broken fluorescents).
float harsh = step(0.92, h3);
flicker = mix(flicker, 0.4 + 0.6 * step(0.5, fract(uFacadeTime * 7.0 + h1 * 12.0)), harsh);

vec3 warmColor = vec3(1.0, 0.65, 0.25);
vec3 coolColor = vec3(0.22, 0.88, 1.0);
vec3 winColor = mix(warmColor, coolColor, h2);

// Skip windows on top/bottom faces.
float horizontal = step(0.65, absN.y);
float windowMask = lit * window * (1.0 - horizontal);
vec3 emit = winColor * windowMask * flicker * 2.4;

totalEmissiveRadiance += emit;
`;

type Props = {
  color?: string;
  roughness?: number;
  metalness?: number;
  emissive?: string;
  emissiveIntensity?: number;
};

export function FacadeMaterial({
  color = "#0a0a14",
  roughness = 0.78,
  metalness = 0.32,
  emissive = "#070914",
  emissiveIntensity = 0.15,
}: Props) {
  const ref = useRef<MeshStandardMaterial>(null);
  const uniforms = useMemo(() => ({ uFacadeTime: { value: 0 } }), []);

  useFrame((state) => {
    uniforms.uFacadeTime.value = state.clock.elapsedTime;
  });

  return (
    <meshStandardMaterial
      ref={ref}
      color={color}
      roughness={roughness}
      metalness={metalness}
      emissive={emissive}
      emissiveIntensity={emissiveIntensity}
      onBeforeCompile={(shader) => {
        shader.uniforms.uFacadeTime = uniforms.uFacadeTime;
        shader.vertexShader = shader.vertexShader.replace(
          "#include <common>",
          `#include <common>\n${VERTEX_HEADER}`,
        );
        shader.vertexShader = shader.vertexShader.replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>\n${VERTEX_BODY}`,
        );
        shader.fragmentShader = shader.fragmentShader.replace(
          "#include <common>",
          `#include <common>\n${FRAGMENT_HEADER}`,
        );
        shader.fragmentShader = shader.fragmentShader.replace(
          "#include <emissivemap_fragment>",
          `#include <emissivemap_fragment>\n${FRAGMENT_BODY}`,
        );
      }}
    />
  );
}
