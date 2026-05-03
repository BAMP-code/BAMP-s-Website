import { useMemo } from "react";

// Procedural rooftop machinery — AC chillers, vent stacks, water
// tank, comm antennas. Per the research brief, real night-city
// buildings have "rooftop machinery (cooling towers, vent stacks,
// antennas, water tanks). Parapet ledges that self-shadow." Without
// these, every building roofline reads as "perfectly flat box top"
// which is the #1 fake tell.
//
// Pseudo-random placement seeded by `seed` so each building's roof
// looks different but stays consistent across renders.

type Props = {
  seed: number;
  width: number;
  depth: number;
  baseY: number;
};

function frac(n: number) {
  return n - Math.floor(n);
}

function rand(seed: number, salt: number) {
  return frac(Math.sin(seed * 12.9898 + salt * 78.233) * 43758.5453);
}

export function RooftopGreebles({ seed, width, depth, baseY }: Props) {
  // Build a small set of greeble specs once — randomized but deterministic.
  const greebles = useMemo(() => {
    const items: Array<{
      kind: "ac" | "vent" | "tank" | "antenna" | "panel";
      x: number;
      z: number;
      sx: number;
      sy: number;
      sz: number;
      color: string;
    }> = [];

    // Parapet wall around the roof edge.
    // (rendered separately below so it doesn't share placement RNG)

    // 2–3 AC unit clusters.
    const acCount = 2 + Math.floor((rand(seed, 1)) * 2);
    for (let i = 0; i < acCount; i++) {
      items.push({
        kind: "ac",
        x: ((rand(seed, 2 + i * 3)) - 0.5) * (width * 0.55),
        z: ((rand(seed, 3 + i * 3)) - 0.5) * (depth * 0.55),
        sx: 0.7 + (rand(seed, 4 + i * 3)) * 0.6,
        sy: 0.45 + (rand(seed, 5 + i * 3)) * 0.35,
        sz: 0.6 + (rand(seed, 6 + i * 3)) * 0.5,
        color: "#1a1a22",
      });
    }

    // 1 water tank (cylindrical-shaped via tall thin box for now).
    if ((rand(seed, 11)) > 0.4) {
      items.push({
        kind: "tank",
        x: ((rand(seed, 12)) - 0.5) * (width * 0.4),
        z: ((rand(seed, 13)) - 0.5) * (depth * 0.4),
        sx: 0.85,
        sy: 1.4 + (rand(seed, 14)) * 0.6,
        sz: 0.85,
        color: "#1a1a22",
      });
    }

    // 1–2 vent stacks (thin tall boxes).
    const ventCount = 1 + Math.floor((rand(seed, 21)) * 2);
    for (let i = 0; i < ventCount; i++) {
      items.push({
        kind: "vent",
        x: ((rand(seed, 22 + i * 4)) - 0.5) * (width * 0.6),
        z: ((rand(seed, 23 + i * 4)) - 0.5) * (depth * 0.6),
        sx: 0.18,
        sy: 0.9 + (rand(seed, 24 + i * 4)) * 0.5,
        sz: 0.18,
        color: "#0e0e16",
      });
    }

    return items;
  }, [seed, width, depth]);

  // Antenna(s) — separate so we can render thin tall cylinders.
  const antennas = useMemo(() => {
    const out: Array<{
      x: number;
      z: number;
      h: number;
      tipColor: string;
    }> = [];
    const count = 1 + Math.floor((rand(seed, 31)) * 2);
    for (let i = 0; i < count; i++) {
      out.push({
        x: ((rand(seed, 32 + i * 3)) - 0.5) * (width * 0.7),
        z: ((rand(seed, 33 + i * 3)) - 0.5) * (depth * 0.7),
        h: 1.4 + (rand(seed, 34 + i * 3)) * 1.4,
        tipColor: (rand(seed, 35 + i * 3)) > 0.5 ? "#ff2222" : "#ffffff",
      });
    }
    return out;
  }, [seed, width, depth]);

  return (
    <group position={[0, baseY, 0]}>
      {/* Parapet — a low wall ringing the rooftop. */}
      <mesh position={[0, 0.18, depth * 0.5 - 0.08]}>
        <boxGeometry args={[width, 0.36, 0.16]} />
        <meshStandardMaterial color="#0a0a14" roughness={0.85} metalness={0.1} />
      </mesh>
      <mesh position={[0, 0.18, -depth * 0.5 + 0.08]}>
        <boxGeometry args={[width, 0.36, 0.16]} />
        <meshStandardMaterial color="#0a0a14" roughness={0.85} metalness={0.1} />
      </mesh>
      <mesh position={[width * 0.5 - 0.08, 0.18, 0]}>
        <boxGeometry args={[0.16, 0.36, depth]} />
        <meshStandardMaterial color="#0a0a14" roughness={0.85} metalness={0.1} />
      </mesh>
      <mesh position={[-width * 0.5 + 0.08, 0.18, 0]}>
        <boxGeometry args={[0.16, 0.36, depth]} />
        <meshStandardMaterial color="#0a0a14" roughness={0.85} metalness={0.1} />
      </mesh>

      {/* Greebles. */}
      {greebles.map((g, i) => (
        <mesh key={`greeble-${i}`} position={[g.x, 0.36 + g.sy * 0.5, g.z]}>
          <boxGeometry args={[g.sx, g.sy, g.sz]} />
          <meshStandardMaterial
            color={g.color}
            roughness={0.8}
            metalness={0.35}
          />
        </mesh>
      ))}

      {/* Antennas with blinking tip beacons. */}
      {antennas.map((a, i) => (
        <group key={`ant-${i}`} position={[a.x, 0, a.z]}>
          <mesh position={[0, 0.36 + a.h / 2, 0]}>
            <cylinderGeometry args={[0.04, 0.04, a.h, 6]} />
            <meshStandardMaterial color="#28283c" metalness={0.55} />
          </mesh>
          <mesh position={[0, 0.36 + a.h + 0.06, 0]}>
            <sphereGeometry args={[0.07, 8, 8]} />
            <meshStandardMaterial
              color={a.tipColor}
              emissive={a.tipColor}
              emissiveIntensity={2.4}
              toneMapped={false}
            />
          </mesh>
        </group>
      ))}
    </group>
  );
}
