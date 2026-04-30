import { useFrame, useThree } from "@react-three/fiber";
import { scrollProgressRef } from "@/lib/scroll";

// Camera Y descends from altitude (Y_START) to street level (Y_END)
// as scroll progresses 0 → 1. Buildings around the corridor are 28–48
// units tall, so we start above their tops looking down and finish at
// roughly head height looking up. Ease-out so the descent slows near
// the street, matching the storyboard in REFACTOR_VISION.md §1.
const Y_START = 32;
const Y_END = 2.2;

export function CameraRig() {
  const { camera } = useThree();

  useFrame(() => {
    const p = scrollProgressRef.current;
    const t = 1 - Math.pow(1 - p, 2);
    camera.position.y = Y_START + (Y_END - Y_START) * t;
    // Look forward and down — pitch eases from "looking down at the
    // city" to "looking forward at street level" as we descend.
    const lookY = camera.position.y - 6 + t * 5;
    camera.lookAt(0, lookY, -22);
  });

  return null;
}
