// CRT/LED screen shader. Applied to every project screen so stills
// and video alike share one visual language: scanlines, RGB
// chromatic split, a scrolling refresh bar, micro-jitter, and an
// edge vignette. All effects are subtle individually; together they
// make a flat poster read as "live" hardware.

export const screenVertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const screenFragmentShader = `
varying vec2 vUv;
uniform sampler2D uMap;
uniform float uTime;

float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
  // Horizontal micro-jitter, time-quantized so it has a visible cadence.
  float jitter = (random(vec2(floor(vUv.y * 220.0), floor(uTime * 60.0))) - 0.5) * 0.0018;
  vec2 uv = vUv + vec2(jitter, 0.0);

  // RGB chromatic split: shift red right, blue left.
  float split = 0.003;
  float r = texture2D(uMap, uv + vec2(split, 0.0)).r;
  float g = texture2D(uMap, uv).g;
  float b = texture2D(uMap, uv - vec2(split, 0.0)).b;
  vec3 color = vec3(r, g, b);

  // Scanlines.
  float scan = sin(vUv.y * 420.0) * 0.07;
  color -= scan;

  // Vertical refresh bar that scrolls top → bottom.
  float barPos = fract(vUv.y - uTime * 0.18);
  float bar = smoothstep(0.06, 0.0, abs(barPos - 0.5)) * 0.16;
  color += vec3(0.35, 0.9, 1.0) * bar;

  // Edge vignette.
  vec2 cv = vUv - 0.5;
  float vignette = smoothstep(0.85, 0.4, length(cv));
  color *= vignette;

  gl_FragColor = vec4(color, 1.0);
}
`;
