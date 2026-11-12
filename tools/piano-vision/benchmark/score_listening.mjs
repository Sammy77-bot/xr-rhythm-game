import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

const [manifestPath, output, modulePath, split] = process.argv.slice(2);
if (!manifestPath || !output) {
  throw new Error("Usage: node score_listening.mjs manifest.json output.json [detector-module] [valid|test]");
}
const source = modulePath ? pathToFileURL(resolve(modulePath)).href : new URL("../pitch.js", import.meta.url);
const { createPitchDetector } = await import(source);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const root = dirname(resolve(manifestPath));
const rate = 48000;
const windowSize = 4096;
const hop = Math.round(0.028 * rate);

function loadAudio(filename) {
  const run = spawnSync("ffmpeg", [
    "-nostdin", "-v", "error", "-i", filename, "-t", String(manifest.seconds_per_case),
    "-ac", "1", "-ar", String(rate), "-f", "f32le", "-",
  ], { maxBuffer: 32 * 1024 * 1024 });
  if (run.status !== 0) throw new Error(run.error?.message || run.stderr.toString());
  return new Float32Array(run.stdout.buffer, run.stdout.byteOffset, run.stdout.length / 4);
}

function loadNotes(filename) {
  return readFileSync(filename, "utf8").split(/\r?\n/).filter(line => line && !line.startsWith("#"))
    .map(line => line.split("\t").map(Number))
    .map(([start, release, end, midi]) => ({ start, release, end, midi }));
}

function tally(result, target, detected) {
  result.total += 1;
  if (detected === target) result.correct += 1;
  else if (detected === null) result.missed += 1;
  else if ((detected - target) % 12 === 0) result.octave += 1;
  else result.wrong += 1;
}

function counts() {
  return { total: 0, correct: 0, missed: 0, octave: 0, wrong: 0 };
}

function percentile(values, fraction) {
  return values.length ? [...values].sort((a, b) => a - b)[Math.ceil(fraction * values.length) - 1] : null;
}

const results = [];
for (const entry of manifest.cases.filter(item => !split || item.split === split)) {
  const audio = loadAudio(resolve(root, "media", entry.id + ".wav"));
  const notes = loadNotes(resolve(root, "labels", entry.id + ".tsv"));
  const detector = createPitchDetector(rate, windowSize);
  const clean = counts(), attack = counts(), overlap = { total: 0, anyCorrect: 0 };
  const byBand = { low: counts(), mid: counts(), high: counts() };
  const confusions = {};
  const confusionExamples = {};
  const timing = [];
  let confirmed = null, previous = null;
  const firstCorrect = new Map();
  const eligible = new Set();
  for (let end = windowSize; end <= audio.length; end += hop) {
    const time = end / rate;
    const started = performance.now();
    const midi = detector(audio.subarray(end - windowSize, end))?.midi ?? null;
    timing.push(performance.now() - started);
    if (midi === previous) confirmed = midi;
    previous = midi;
    const sounding = notes.filter(note => note.start <= time && note.end > time);
    const held = notes.filter(note => note.start <= time && note.release > time);
    if (held.length > 1) {
      overlap.total += 1;
      overlap.anyCorrect += Number(held.some(note => note.midi === confirmed));
    }
    if (held.length !== 1 || held[0].midi < 33 || held[0].midi > 96) continue;
    const target = held[0];
    const age = time - target.start;
    if (age < 0.10) tally(attack, target.midi, confirmed);
    if (target.release - target.start >= 0.14 && sounding.length === 1 &&
        !notes.some(note => note !== target && note.midi === target.midi &&
          note.start < target.start && note.end > target.start - 0.12)) {
      eligible.add(target);
      if (confirmed === target.midi && !firstCorrect.has(target)) firstCorrect.set(target, age * 1000);
    }
    if (age < 0.10 || sounding.length !== 1 ||
        notes.some(note => note !== target && note.start < time && note.end > time - 0.10)) continue;
    tally(clean, target.midi, confirmed);
    tally(byBand[target.midi < 53 ? "low" : target.midi < 76 ? "mid" : "high"], target.midi, confirmed);
    if (confirmed !== target.midi) {
      const key = `${target.midi}->${confirmed ?? "none"}`;
      confusions[key] = (confusions[key] ?? 0) + 1;
      confusionExamples[key] ??= {
        time_s: Number(time.toFixed(3)),
        note_age_ms: Math.round(age * 1000),
      };
    }
  }
  const result = {
    ...entry, clean, attack, overlap, byBand, confusions, confusionExamples,
    updateMs: { median: percentile(timing.slice(20), 0.5), p95: percentile(timing.slice(20), 0.95) },
    firstCorrect: { total: eligible.size, found: firstCorrect.size, median_ms: percentile([...firstCorrect.values()], 0.5) },
  };
  results.push(result);
  console.log(JSON.stringify(result));
}
writeFileSync(output, JSON.stringify({ windowSize, sampleRate: rate, hop, results }, null, 2) + "\n");
