import { useFrame, useThree } from "@react-three/fiber";
import { scrollProgressRef } from "@/lib/scroll";

// Camera Y descends from altitude (Y_START) to street level (Y_END)
// as scroll progresses 0 → 1. Ease-out so the descent slows near the
// street, matching the storyboard in REFACTOR_VISION.md §1.
const Y_START = 12;
const Y_END = 1.5;

export function CameraRig() {
  const { camera } = useThree();

  useFrame(() => {
    const p = scrollProgressRef.current;
    const t = 1 - Math.pow(1 - p, 2);
    camera.position.y = Y_START + (Y_END - Y_START) * t;
    camera.lookAt(0, camera.position.y - 1, -10);
  });

  return null;
}
