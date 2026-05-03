import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import { RepeatWrapping } from "three";
import type { MeshStandardMaterial, Texture } from "three";

// Procedural emissive window-grid material on top of real PBR
// concrete-wall textures (Poly Haven CC0 `concrete_wall_007`). The
// PBR base provides micro-detail (panel seams, surface variation,
// cracks) so the building reads as actual concrete; the shader
// overlay adds the window-grid emissives.
//
// The shader is injected into MeshStandardMaterial via onBeforeCompile
// so PBR + custom emissive coexist. Per-cell hashing drives on/off
// (~55% lit), warm/cool color mix (sodium-amber ↔ tech-cyan), and a
// slow per-window flicker. Triplanar projection in world space keeps
// window aspect square on each face. Top/bottom faces skipped.
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

vec2 cellSize = vec2(0.55, 0.9);
vec2 cell = floor(gridUV / cellSize);
vec2 cellLocal = fract(gridUV / cellSize);

vec2 winInset = vec2(0.18, 0.13);
float window =
  step(winInset.x, cellLocal.x) * step(cellLocal.x, 1.0 - winInset.x) *
  step(winInset.y, cellLocal.y) * step(cellLocal.y, 1.0 - winInset.y);

float h1 = hash21(cell);
float h2 = hash21(cell + vec2(7.1, 13.7));
float h3 = hash21(cell + vec2(31.5, 57.3));
float lit = step(0.55, h1);
float flicker = 0.78 + 0.22 * sin(uFacadeTime * (1.5 + h2 * 3.0) + h1 * 6.28);
float harsh = step(0.92, h3);
flicker = mix(flicker, 0.4 + 0.6 * step(0.5, fract(uFacadeTime * 7.0 + h1 * 12.0)), harsh);

vec3 warmColor = vec3(1.0, 0.65, 0.25);
vec3 coolColor = vec3(0.22, 0.88, 1.0);
vec3 winColor = mix(warmColor, coolColor, h2);

float horizontal = step(0.65, absN.y);
float windowMask = lit * window * (1.0 - horizontal);
vec3 emit = winColor * windowMask * flicker * 2.4;

// Darken the diffuse where windows are lit so the texture doesn't
// peek through and wash out the emissive.
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.02), windowMask);

totalEmissiveRadiance += emit;
`;

type Props = {
  color?: string;
  roughness?: number;
  metalness?: number;
  emissive?: string;
  emissiveIntensity?: number;
  /** Enable per-instance color tinting (used by InstancedMesh fillers). */
  vertexColors?: boolean;
  /** Texture tiling repeat (per face). Higher = smaller texture cells. */
  textureRepeat?: number;
};

const facadeTextures: [string, string, string] = [
  "/textures/facade/facade_diff.jpg",
  "/textures/facade/facade_nor.jpg",
  "/textures/facade/facade_rough.jpg",
];

export function FacadeMaterial({
  color = "#1a1a22",
  roughness = 0.85,
  metalness = 0.18,
  emissive = "#070914",
  emissiveIntensity = 0.15,
  vertexColors = false,
  textureRepeat = 4,
}: Props) {
  const ref = useRef<MeshStandardMaterial>(null);
  const [diffMap, norMap, roughMap] = useTexture(facadeTextures) as Texture[];
  const uniforms = useMemo(() => ({ uFacadeTime: { value: 0 } }), []);

  // Configure tiling once. Same instances are reused across renders
  // by drei, so this stays cheap.
  useEffect(() => {
    [diffMap, norMap, roughMap].forEach((t) => {
      t.wrapS = RepeatWrapping;
      t.wrapT = RepeatWrapping;
      t.repeat.set(textureRepeat, textureRepeat);
      t.needsUpdate = true;
    });
  }, [diffMap, norMap, roughMap, textureRepeat]);

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
      vertexColors={vertexColors}
      map={diffMap}
      normalMap={norMap}
      roughnessMap={roughMap}
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
