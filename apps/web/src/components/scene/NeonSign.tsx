import { Text } from "@react-three/drei";

// Modeled neon sign — backing frame + channel-letter face + a paired
// off-screen point light that washes the building behind. Per the
// research brief: "the wall behind a sign is washed in the sign's
// color — this is the #1 thing readers subconsciously check for. The
// trick is it's not the sign that lights the wall, it's a paired
// hidden light."
//
// Mounted perpendicular to building faces so the camera reads them as
// real protruding signage (Mongkok / Shibuya / Akihabara).

type Props = {
  text: string;
  position: [number, number, number];
  rotation?: [number, number, number];
  color?: string;
  scale?: number;
};

export function NeonSign({
  text,
  position,
  rotation = [0, 0, 0],
  color = "#ff2bd6",
  scale = 1,
}: Props) {
  const frameWidth = text.length * 0.55 * scale + 0.4 * scale;
  const frameHeight = 1.1 * scale;

  return (
    <group position={position} rotation={rotation}>
      {/* Backing frame — slightly behind the text so the letters
          appear to sit on a panel. */}
      <mesh position={[0, 0, -0.02]}>
        <boxGeometry args={[frameWidth, frameHeight, 0.08]} />
        <meshStandardMaterial
          color="#0a0a14"
          roughness={0.55}
          metalness={0.55}
          emissive="#1a1a26"
          emissiveIntensity={0.3}
        />
      </mesh>

      {/* Mounting brackets — visible structure on each side. */}
      <mesh position={[-frameWidth / 2 - 0.08, 0, -0.06]}>
        <boxGeometry args={[0.12, frameHeight * 0.8, 0.16]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} roughness={0.5} />
      </mesh>
      <mesh position={[frameWidth / 2 + 0.08, 0, -0.06]}>
        <boxGeometry args={[0.12, frameHeight * 0.8, 0.16]} />
        <meshStandardMaterial color="#28283c" metalness={0.6} roughness={0.5} />
      </mesh>

      {/* Channel-letter face — emissive text, intensity past 1 so the
          tightened bloom catches it. */}
      <Text
        position={[0, 0, 0.05]}
        fontSize={0.7 * scale}
        color={color}
        outlineColor={color}
        outlineWidth={0.022 * scale}
        outlineOpacity={0.75}
        anchorX="center"
        anchorY="middle"
      >
        {text}
      </Text>

      {/* Paired hidden light — the trick that makes the sign feel real.
          Sits in front of the sign so it spills onto the building behind. */}
      <pointLight
        position={[0, 0, 1.6 * scale]}
        color={color}
        intensity={3.0 * scale}
        distance={9 * scale}
      />
    </group>
  );
}
