import { bendPoint, createKeyNet, createKeyNetSession, keyOutlines } from "pianocv";
import ortGpuWasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import { createPitchDetector, noteName } from "./pitch.js";
import { createRectifier, xrealLens } from "./xreal-lens.js";

const get = (id) => document.getElementById(id);
const canvas = get("output");
const ctx = canvas.getContext("2d");
const noteCanvas = get("note-overlay");
const noteCtx = noteCanvas.getContext("2d");
const sensor = document.createElement("canvas");
const sensorCtx = sensor.getContext("2d");
const oriented = document.createElement("canvas");
const orientedCtx = oriented.getContext("2d");
let active;
let latest;
let model;
let microphone;
let playedNote;

function setStatus(message) {
  get("status").textContent = message;
}

function clearResult() {
  latest = null;
  noteCanvas.hidden = true;
  get("counts").textContent = "";
  get("export").disabled = true;
}

function renderPlayedNote() {
  noteCanvas.hidden = true;
  if (!playedNote) return;
  const input = get("midi");
  const base = input.valueAsNumber;
  const key = latest && input.checkValidity() &&
    [0, 2, 4, 5, 7, 9, 11].includes(base % 12) &&
    latest.keys.find((entry) => base + entry.semitone === playedNote.midi);
  const note = `${noteName(playedNote.midi)} · MIDI ${playedNote.midi} · ${playedNote.hz.toFixed(1)} Hz`;
  get("played-note").textContent = key ? note : `${note} · 未匹配画面键位`;
  if (!key) return;
  const { width, height } = latest;
  if (noteCanvas.width !== width) noteCanvas.width = width;
  if (noteCanvas.height !== height) noteCanvas.height = height;
  noteCtx.clearRect(0, 0, width, height);
  noteCtx.beginPath();
  for (const [i, point] of key.top.entries()) {
    noteCtx[i ? "lineTo" : "moveTo"](point.x * width, point.y * height);
  }
  noteCtx.closePath();
  noteCtx.fillStyle = "#65f7c4b0";
  noteCtx.strokeStyle = "#f4fff9";
  noteCtx.lineWidth = Math.max(1, width / 600);
  noteCtx.fill();
  noteCtx.stroke();
  noteCanvas.hidden = false;
}

function stopMicrophone() {
  const job = microphone;
  microphone = undefined;
  clearInterval(job?.timer);
  job?.stream?.getTracks().forEach((track) => track.stop());
  void job?.audio?.close().catch(() => {});
  playedNote = undefined;
  noteCanvas.hidden = true;
  get("listen").textContent = "开启麦克风听音";
  get("listen").setAttribute("aria-pressed", "false");
  get("played-note").textContent = "单音识别 · 等待开启";
}

get("listen").onclick = async () => {
  if (microphone) {
    stopMicrophone();
    return;
  }
  const job = {};
  microphone = job;
  get("listen").textContent = "停止麦克风";
  get("listen").setAttribute("aria-pressed", "true");
  get("played-note").textContent = "等待麦克风权限…";
  try {
    job.audio = new AudioContext();
    await job.audio.resume();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    if (microphone !== job) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    job.stream = stream;
    const analyser = job.audio.createAnalyser();
    analyser.fftSize = 4096;
    job.audio.createMediaStreamSource(stream).connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    const detect = createPitchDetector(job.audio.sampleRate, samples.length);
    let previous;
    get("played-note").textContent = "正在听音 · 请弹一个音";
    job.timer = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      const pitch = detect(samples);
      if (pitch?.midi === previous) {
        playedNote = pitch;
        if (!pitch) get("played-note").textContent = "正在听音 · 请弹一个音";
        renderPlayedNote();
      }
      previous = pitch?.midi;
    }, 28);
  } catch (error) {
    if (microphone !== job) return;
    stopMicrophone();
    get("played-note").textContent = `麦克风不可用 · ${error.message}`;
  }
};
get("midi").addEventListener("input", renderPlayedNote);

function showFrame(frame) {
  if (canvas.width !== frame.width) canvas.width = frame.width;
  if (canvas.height !== frame.height) canvas.height = frame.height;
  ctx.drawImage(frame, 0, 0);
  canvas.hidden = false;
  get("placeholder").hidden = true;
}

function stop() {
  active?.controller.abort();
  active?.stream?.getTracks().forEach((track) => track.stop());
  if (active?.video) active.video.srcObject = null;
  active = null;
  clearResult();
  canvas.hidden = true;
  get("placeholder").hidden = false;
  get("credit").hidden = true;
  get("source").textContent = "未选择输入";
  setStatus("等待画面");
}

async function start(source, title, run) {
  stop();
  const job = { source, controller: new AbortController() };
  active = job;
  get("source").textContent = title;
  setStatus("正在加载本地识别模型…");
  try {
    model ??= createKeyNet({ ortGpuWasm }, "/keynet.onnx").catch((error) => {
      model = undefined;
      throw error;
    });
    job.runner = await model;
    if (active !== job) return;
    job.session = createKeyNetSession(job.runner);
    await run(job);
  } catch (error) {
    if (active !== job) return;
    stop();
    setStatus(error.message || String(error));
  }
}

async function identify(job, frame, stamp = null, rotation = 0, started = performance.now()) {
  const size = { width: frame.width, height: frame.height };
  const result = await job.session.step(frame, size, performance.now());
  if (active !== job) return false;
  if (job.source === "xrgate" && (rotation !== Number(get("rotation").value) ||
      job.rectifiedRequested !== get("xreal-rectify").checked)) return false;
  const fit = result?.fit;
  const keys = fit ? keyOutlines(fit) : [];
  showFrame(frame);
  clearResult();
  if (!keys.length) {
    job.lastUpdate = undefined;
    if (job.source === "xrgate") {
      const elapsed = ((performance.now() - job.searchStarted) / 1000).toFixed(1);
      setStatus(`寻找键盘 · 已搜索 ${elapsed} 秒 · ${job.runner.backend}`);
    } else {
      setStatus(`寻找键盘 · ${job.runner.backend}`);
    }
    return false;
  }
  let whiteCount = 0;
  for (const key of keys) {
    if (!key.black) whiteCount += 1;
    ctx.beginPath();
    for (const [index, point] of key.top.entries()) {
      const method = index === 0 ? "moveTo" : "lineTo";
      ctx[method](point.x * size.width, point.y * size.height);
    }
    ctx.closePath();
    ctx.strokeStyle = key.black ? "#ffc286" : "#8bffc1";
    ctx.fillStyle = key.black ? "#ffc28638" : "#8bffc11d";
    ctx.lineWidth = Math.max(1, size.width / 700);
    ctx.fill();
    ctx.stroke();
  }
  latest = { source: job.source, ...size, stamp, rotation, fit, keys, rectified: job.rectified };
  renderPlayedNote();
  const expected = job.source === "xrgate" && Number(get("xreal-keys").value);
  const blackCount = keys.length - whiteCount;
  const expectedWhite = expected === 61 ? 36 : 52;
  const countCheck = !expected ? "" : whiteCount === expectedWhite && keys.length === expected
    ? ` · 符合 ${expected} 键` : ` · 与 ${expected} 键规格不符`;
  get("counts").textContent = `${whiteCount} 白键 / ${blackCount} 黑键${countCheck}`;
  get("export").disabled = false;
  const finished = performance.now();
  const interval = job.lastUpdate === undefined ? "" :
    ` · 结果间隔 ${Math.round(finished - job.lastUpdate)} ms`;
  job.lastUpdate = finished;
  if (job.source === "xrgate") job.lockMs ??= Math.round(finished - job.searchStarted);
  const lock = job.lockMs === undefined ? "" : ` · 首次定位 ${job.lockMs} ms`;
  setStatus(`已拟合 · ${job.runner.backend} · 帧处理 ${Math.round(finished - started)} ms${interval}${lock}`);
  return true;
}

function decodeGray8(buffer) {
  if (buffer.byteLength < 16) throw new Error("XREAL 帧头不完整");
  const header = new DataView(buffer);
  const width = header.getUint32(0, true);
  const height = header.getUint32(4, true);
  if (!width || !height || width > 4096 || height > 4096 ||
      buffer.byteLength !== 16 + width * height) {
    throw new Error("XREAL 灰度帧尺寸与载荷不符");
  }
  return {
    width,
    height,
    stamp: header.getBigUint64(8, true).toString(),
    pixels: new Uint8Array(buffer, 16),
  };
}

let image;
let rectifyGray;

function drawGray8({ width, height, pixels }, rotation, rectified) {
  if (rectified) {
    rectifyGray ??= createRectifier(xrealLens);
    pixels = rectifyGray(pixels);
  }
  if (!image || image.width !== width || image.height !== height) {
    sensor.width = width;
    sensor.height = height;
    image = sensorCtx.createImageData(width, height);
  }
  for (let i = 0; i < pixels.length; i += 1) {
    const offset = i * 4;
    image.data[offset] = pixels[i];
    image.data[offset + 1] = pixels[i];
    image.data[offset + 2] = pixels[i];
    image.data[offset + 3] = 255;
  }
  sensorCtx.putImageData(image, 0, 0);
  const swap = rotation === 90 || rotation === 270;
  const viewWidth = swap ? height : width;
  const viewHeight = swap ? width : height;
  if (oriented.width !== viewWidth) oriented.width = viewWidth;
  if (oriented.height !== viewHeight) oriented.height = viewHeight;
  const radians = rotation * Math.PI / 180;
  const cos = Math.round(Math.cos(radians));
  const sin = Math.round(Math.sin(radians));
  orientedCtx.setTransform(cos, sin, -sin, cos, viewWidth / 2, viewHeight / 2);
  orientedCtx.drawImage(sensor, -width / 2, -height / 2);
  return oriented;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fromXreal(job) {
  let lastStamp;
  let lastRotation;
  let lastFresh = 0;
  while (active === job) {
    let frame;
    try {
      const signal = AbortSignal.any([job.controller.signal, AbortSignal.timeout(1800)]);
      const response = await fetch("/xr-frame", { signal, cache: "no-store" });
      if (!response.ok) throw new Error("请先启动 XR-Gate 和相机桥接");
      frame = decodeGray8(await response.arrayBuffer());
    } catch (error) {
      if (active !== job) return;
      if (lastStamp !== undefined) job.session = createKeyNetSession(job.runner);
      lastStamp = undefined;
      job.lastUpdate = undefined;
      clearResult();
      canvas.hidden = true;
      get("placeholder").hidden = false;
      setStatus(`XREAL 相机无可用帧 · ${error.message}`);
      await sleep(500);
      continue;
    }
    if (active !== job) return;
    const rotation = Number(get("rotation").value);
    const now = performance.now();
    const fresh = frame.stamp !== lastStamp;
    const gap = now - lastFresh;
    if (fresh) lastFresh = now;
    if (!fresh && now - lastFresh > 1200) {
      job.lastUpdate = undefined;
      clearResult();
      canvas.hidden = true;
      get("placeholder").hidden = false;
      setStatus("XREAL 相机画面停止更新");
    } else if (fresh || rotation !== lastRotation || job.rectifiedRequested !== get("xreal-rectify").checked) {
      const requested = get("xreal-rectify").checked;
      if (rotation !== lastRotation || gap > 1200 || job.rectifiedRequested !== requested) {
        job.session = createKeyNetSession(job.runner);
        job.lastUpdate = undefined;
        job.searchStarted = now;
        job.lockMs = undefined;
      }
      lastStamp = frame.stamp;
      lastRotation = rotation;
      job.rectifiedRequested = requested;
      job.rectified = requested && frame.width === xrealLens.width && frame.height === xrealLens.height;
      await identify(job, drawGray8(frame, rotation, job.rectified), frame.stamp, rotation, now);
      continue;
    }
    await sleep(16);
  }
}

async function fromWebcam(job) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  if (active !== job) {
    stream.getTracks().forEach((track) => track.stop());
    return;
  }
  const video = document.createElement("video");
  job.stream = stream;
  job.video = video;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  const frame = document.createElement("canvas");
  const frameCtx = frame.getContext("2d");
  let lastTime;
  let lastFresh = performance.now();
  while (active === job) {
    if (video.videoWidth && video.currentTime !== lastTime && video.readyState >= 2) {
      if (performance.now() - lastFresh > 1200) job.session = createKeyNetSession(job.runner);
      lastTime = video.currentTime;
      lastFresh = performance.now();
      if (frame.width !== video.videoWidth) frame.width = video.videoWidth;
      if (frame.height !== video.videoHeight) frame.height = video.videoHeight;
      frameCtx.drawImage(video, 0, 0);
      await identify(job, frame, null, 0, lastFresh);
      continue;
    } else if (performance.now() - lastFresh > 1200) {
      job.lastUpdate = undefined;
      clearResult();
      canvas.hidden = true;
      get("placeholder").hidden = false;
      setStatus("摄像头画面停止更新");
    }
    await sleep(16);
  }
}

async function fromImage(job, blob) {
  const image = await createImageBitmap(blob);
  try {
    // The upstream tracker votes across multiple frames before accepting a layout.
    for (let attempt = 0; attempt < 10 && active === job; attempt += 1) {
      if (await identify(job, image)) return;
    }
    if (active === job) setStatus("未找到键盘 · 试试拍到完整键盘和前沿");
  } finally {
    image.close();
  }
}

get("xr").onclick = () => start("xrgate", "XREAL / camera0", fromXreal);
get("camera").onclick = () => start("webcam", "普通摄像头", fromWebcam);
get("stop").onclick = stop;
get("open-image").onclick = () => get("image").click();
get("image").onchange = (event) => {
  const file = event.target.files[0];
  if (!file) return;
  event.target.value = "";
  void start("image", file.name, (job) => fromImage(job, file));
};
get("sample").onclick = () => start("image", "示例 / Yamaha P-140S", async (job) => {
  get("credit").hidden = false;
  const response = await fetch("/sample-piano.jpg", { signal: job.controller.signal });
  if (!response.ok) throw new Error("示例照片读取失败");
  await fromImage(job, await response.blob());
});

get("fullscreen").onclick = async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await get("monitor").requestFullscreen();
};
get("rotation").onchange = () => {
  if (active?.source !== "xrgate") return;
  clearResult();
  canvas.hidden = true;
  get("placeholder").hidden = false;
  setStatus("正在调整画面方向…");
};

get("export").onclick = () => {
  if (!latest) return;
  const input = get("midi");
  const base = input.value === "" ? null : input.valueAsNumber;
  const { source, width, height, stamp, rotation, fit, keys } = latest;
  const isWhite = base === null || [0, 2, 4, 5, 7, 9, 11].includes(base % 12);
  if (base !== null && (!input.checkValidity() || !isWhite ||
      keys.some((key) => base + key.semitone < 0 || base + key.semitone > 127))) {
    setStatus("请填写有效的最低白键 MIDI 编号，且所有识别键须在 0–127 范围内");
    return;
  }
  const data = {
    version: 1,
    source,
    frame: { width, height, timestamp_ns: stamp, clockwise_rotation_deg: rotation,
      xreal_fisheye_rectified: latest.rectified ?? false },
    coordinates: "normalized_image_top_left",
    keyboard: {
      phase: fit.phase,
      first_white_midi: base,
      quad: fit.quad.map((point) => bendPoint(fit.lens, point)),
    },
    keys: keys.map((key) => ({
      black: key.black,
      semitones_from_first_white: key.semitone,
      midi: base === null ? null : base + key.semitone,
      polygon: key.top,
    })),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "piano-keys.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
window.addEventListener("pagehide", () => {
  stopMicrophone();
  stop();
});
