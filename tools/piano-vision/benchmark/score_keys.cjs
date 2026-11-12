// Exercise the same image inference and export path as the desktop demo.
const { spawnSync } = require("node:child_process");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

function extractFrame(video, timestamp, mode, target, { rotation_deg: rotation = 0, crop_px: crop }) {
  if (rotation !== 0 && rotation !== 180) throw new Error("Frame rotation must be 0 or 180 degrees");
  const filters = [];
  if (rotation === 180) filters.push("hflip", "vflip");
  if (crop) {
    if (!Array.isArray(crop) || crop.length !== 4 ||
        !crop.every((value, index) => Number.isInteger(value) && value >= (index < 2 ? 1 : 0))) {
      throw new Error("Crop must be [width, height, x, y] in source pixels");
    }
    filters.push(`crop=${crop.join(":")}`);
  }
  if (mode === "gray") filters.push("format=gray");
  const filter = filters.length ? ["-vf", filters.join(",")] : [];
  const result = spawnSync("ffmpeg", [
    "-nostdin", "-loglevel", "error", "-y", "-ss", String(timestamp),
    "-i", video, "-frames:v", "1", ...filter, target,
  ], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.error?.message || result.stderr);
}

function cornerError(predicted, reference, width, height) {
  if (!reference) return null;
  if (width !== reference.width || height !== reference.height) {
    throw new Error("Reference and frame dimensions differ");
  }
  const label = reference.quad_px;
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const span = (distance(label[0], label[1]) + distance(label[3], label[2])) / 2;
  return 100 * predicted.reduce((sum, p, i) =>
    sum + distance([p.x * width, p.y * height], label[i]), 0) / (4 * span);
}

function containsPoint(polygon, x, y) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i], b = polygon[j];
    if ((a.y > y) !== (b.y > y) &&
        x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

async function main() {
  const [manifestPath, outputPath, clipId] = process.argv.slice(2);
  if (!manifestPath || !outputPath) throw new Error("Usage: node score_keys.cjs manifest.json output.json [clip-id]");
  const root = path.dirname(path.resolve(manifestPath));
  const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), "piano-frames-"));
  const browser = await chromium.launch({ executablePath: process.env.BENCH_BROWSER });
  try {
    const page = await browser.newPage({ acceptDownloads: true });
    await page.goto(process.env.PIANO_DEMO_URL || "http://127.0.0.1:5275/");
    const results = [];
    for (const clip of manifest.clips) {
      if (!clip.video || (clipId && clip.id !== clipId)) continue;
      const video = path.join(root, clip.video);
      for (const mode of ["color", "gray"]) {
        for (const timestamp of clip.frames_s) {
          const target = path.join(temp, `${results.length}.png`);
          extractFrame(video, timestamp, mode, target, clip);
          await page.locator("#stop").click();
          await page.locator("#image").setInputFiles(target);
          await page.waitForFunction(() => {
            const status = document.getElementById("status").textContent;
            return status.startsWith("已拟合") || status.startsWith("未找到键盘");
          }, null, { timeout: 30000 });
          const row = { clip: clip.id, timestamp_s: timestamp, mode, found: false };
          const points = clip.key_points?.[timestamp];
          if (points) row.point_labels = { matched: 0, total: points.length };
          if (await page.locator("#export").isEnabled()) {
            const pending = page.waitForEvent("download");
            await page.locator("#export").click();
            const download = await pending;
            const data = JSON.parse(await fs.readFile(await download.path(), "utf8"));
            row.found = true;
            row.white_keys = data.keys.filter((key) => !key.black).length;
            row.black_keys = data.keys.length - row.white_keys;
            row.exact_count = clip.counts ?
              row.white_keys === clip.counts.white && row.black_keys === clip.counts.black : null;
            row.mean_corner_error_pct_keyboard_width = cornerError(
              data.keyboard.quad, clip.geometry, data.frame.width, data.frame.height,
            );
            row.predicted_quad = data.keyboard.quad;
            if (points) {
              const keys = new Map(data.keys.map((key) => [
                clip.first_white_midi + key.semitones_from_first_white, key,
              ]));
              for (const [midi, x, y] of points) {
                const polygon = keys.get(midi)?.polygon;
                if (polygon && containsPoint(polygon, x / data.frame.width, y / data.frame.height)) {
                  row.point_labels.matched += 1;
                }
              }
            }
          } else {
            row.exact_count = clip.counts ? false : null;
          }
          results.push(row);
          console.log(JSON.stringify(row));
        }
      }
    }
    if (!results.length) throw new Error("No video frames evaluated");
    await fs.writeFile(outputPath, JSON.stringify(results, null, 2) + "\n");
  } finally {
    await browser.close();
    // Only remove the temporary PNGs written by this process.
    for (const name of await fs.readdir(temp)) {
      if (/^\d+\.png$/.test(name)) await fs.unlink(path.join(temp, name));
    }
    await fs.rmdir(temp);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
