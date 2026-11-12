const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

export function noteName(midi) {
  return `${NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function spectralEnergy(signal, weights, period) {
  const coefficient = 2 * Math.cos(2 * Math.PI / period);
  let last = 0;
  let previous = 0;
  const start = signal.length - weights.length;
  for (let i = 0; i < weights.length; i += 1) {
    const next = signal[start + i] * weights[i] + coefficient * last - previous;
    previous = last;
    last = next;
  }
  return last * last + previous * previous - coefficient * last * previous;
}

// YIN cumulative mean normalized difference on a downsampled microphone window.
export function createPitchDetector(sampleRate, windowLength) {
  const stride = Math.max(1, Math.floor(sampleRate / 12000));
  const rate = sampleRate / stride;
  const signal = new Float32Array(Math.floor(windowLength / stride));
  const minLag = Math.max(2, Math.floor(rate / 2600));
  const maxLag = Math.min(Math.ceil(rate / 52), Math.floor(signal.length / 2));
  const overlap = Math.min(signal.length - maxLag, Math.round(rate * 0.045));
  const weights = Float32Array.from({ length: overlap }, (_, i) =>
    0.5 - 0.5 * Math.cos(2 * Math.PI * i / (overlap - 1)));
  const difference = new Float32Array(maxLag + 1);
  return (samples) => {
    let energy = 0;
    for (let i = 0; i < signal.length; i += 1) {
      let sum = 0;
      for (let j = 0; j < stride; j += 1) sum += samples[i * stride + j];
      signal[i] = sum / stride;
      energy += signal[i] ** 2;
    }
    if (energy < signal.length * 0.003 ** 2) return null;
    let cumulative = 0;
    difference[0] = 1;
    for (let lag = 1; lag <= maxLag; lag += 1) {
      let sum = 0;
      // Compare the newest samples for every lag so a previous note contributes less.
      const end = signal.length - lag;
      for (let i = end - overlap; i < end; i += 1) sum += (signal[i] - signal[i + lag]) ** 2;
      cumulative += sum;
      difference[lag] = cumulative > 0 ? sum * lag / cumulative : 1;
    }
    for (let lag = minLag; lag < maxLag; lag += 1) {
      if (difference[lag] >= 0.21) continue;
      while (lag < maxLag && difference[lag + 1] < difference[lag]) lag += 1;
      // A strong second harmonic can make the first dip appear one octave too high.
      const doubled = 2 * lag;
      if (doubled + 2 < maxLag) {
        let alternative = doubled;
        for (let nearby = doubled - 2; nearby <= doubled + 2; nearby += 1) {
          if (difference[nearby] < difference[alternative]) alternative = nearby;
        }
        if (difference[alternative] < difference[lag] * 0.68 &&
            difference[lag] - difference[alternative] > 0.018 &&
            spectralEnergy(signal, weights, alternative) >=
              0.02 * spectralEnergy(signal, weights, alternative / 2)) {
          lag = alternative;
        }
      }
      const left = difference[lag - 1];
      const center = difference[lag];
      const right = difference[lag + 1] ?? center;
      const curvature = left - 2 * center + right;
      const adjustment = curvature > 0 ? Math.max(-0.5, Math.min(0.5, (left - right) / (2 * curvature))) : 0;
      const hz = rate / (lag + adjustment);
      const midi = Math.round(69 + 12 * Math.log2(hz / 440));
      return midi >= 33 && midi <= 96 ? { midi, hz } : null;
    }
    return null;
  };
}
