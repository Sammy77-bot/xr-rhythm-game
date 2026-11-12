// XR-Gate Basalt camera0 calibration for ZBBM5DZFMP, unified_480_ccw90.
export const xrealLens = {
  width: 480, height: 640,
  fx: 238.88264329583697, fy: 238.43988669054954,
  cx: 239.65578750366714, cy: 321.1613847182054,
  k1: 0.004136165249475896, k2: 0.09509478306213548,
  k3: -0.11766502639762973, k4: 0.04622914792374395,
};

export function createRectifier(lens) {
  const { width, height, fx, fy, cx, cy, k1, k2, k3, k4 } = lens;
  const size = width * height;
  const source = new Int32Array(size).fill(-1);
  const wx = new Float32Array(size);
  const wy = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const x = (i % width - cx) / fx;
    const y = (Math.floor(i / width) - cy) / fy;
    const radius = Math.hypot(x, y);
    const theta = Math.atan(radius);
    const squared = theta * theta;
    const scale = radius === 0 ? 1 : theta / radius *
      (1 + squared * (k1 + squared * (k2 + squared * (k3 + squared * k4))));
    const u = cx + fx * x * scale;
    const v = cy + fy * y * scale;
    const left = Math.floor(u);
    const top = Math.floor(v);
    if (left < 0 || top < 0 || left >= width - 1 || top >= height - 1) continue;
    source[i] = top * width + left;
    wx[i] = u - left;
    wy[i] = v - top;
  }
  const output = new Uint8ClampedArray(size);
  return (gray) => {
    for (let i = 0; i < size; i++) {
      const p = source[i];
      if (p < 0) continue;
      const top = gray[p] + wx[i] * (gray[p + 1] - gray[p]);
      const bottom = gray[p + width] + wx[i] * (gray[p + width + 1] - gray[p + width]);
      output[i] = top + wy[i] * (bottom - top);
    }
    return output;
  };
}
