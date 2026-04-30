// Audio system. Rain bed + occasional thunder, all synthesized via
// Web Audio (filtered white noise) so we don't ship audio assets.
// Real samples can drop in later by replacing generateNoiseBuffer
// with decoded audio buffers — the filter chain stays the same.
//
// Policy:
// - Browser autoplay rules require a user gesture before starting.
//   The layout's <script> block calls startAudio() on first
//   click/scroll, then ignores subsequent calls.
// - Mute state persists in localStorage under "bamp:audio:muted".
// - prefers-reduced-motion → mute by default unless the user has
//   explicitly toggled.

const STORAGE_KEY = "bamp:audio:muted";
const RAIN_GAIN = 0.085;

let ctx: AudioContext | null = null;
let rainNode: { source: AudioBufferSourceNode; gain: GainNode } | null = null;
let started = false;
let muted = false;

function loadStoredMute(): boolean | null {
  if (typeof window === "undefined") return null;
  const v = window.localStorage.getItem(STORAGE_KEY);
  if (v === "1") return true;
  if (v === "0") return false;
  return null;
}

function ensureContext(): AudioContext {
  if (!ctx) {
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    ctx = new Ctor();
  }
  if (ctx.state === "suspended") {
    ctx.resume().catch(() => {});
  }
  return ctx;
}

function generateNoiseBuffer(c: AudioContext, durationSeconds = 4): AudioBuffer {
  const sampleRate = c.sampleRate;
  const buffer = c.createBuffer(1, Math.floor(sampleRate * durationSeconds), sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.random() * 2 - 1;
  }
  return buffer;
}

function startRain() {
  const c = ensureContext();
  const buffer = generateNoiseBuffer(c, 4);
  const source = c.createBufferSource();
  source.buffer = buffer;
  source.loop = true;

  // Bandpass-ish: high-pass to remove rumble, low-pass to take edge
  // off the hiss. Together that's a passable rain texture.
  const highpass = c.createBiquadFilter();
  highpass.type = "highpass";
  highpass.frequency.value = 900;
  const lowpass = c.createBiquadFilter();
  lowpass.type = "lowpass";
  lowpass.frequency.value = 6200;

  const gain = c.createGain();
  gain.gain.value = muted ? 0 : RAIN_GAIN;

  source.connect(highpass);
  highpass.connect(lowpass);
  lowpass.connect(gain);
  gain.connect(c.destination);
  source.start();

  rainNode = { source, gain };
}

function playThunder() {
  if (!ctx || muted) return;
  const c = ctx;
  const buffer = generateNoiseBuffer(c, 3.2);
  const source = c.createBufferSource();
  source.buffer = buffer;

  // Heavy low-pass for the rumble.
  const lowpass = c.createBiquadFilter();
  lowpass.type = "lowpass";
  lowpass.frequency.value = 220;
  const gain = c.createGain();
  const now = c.currentTime;
  gain.gain.setValueAtTime(0, now);
  // Quick attack, long decay.
  gain.gain.linearRampToValueAtTime(0.55, now + 0.06);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 3);

  source.connect(lowpass);
  lowpass.connect(gain);
  gain.connect(c.destination);
  source.start(now);
  source.stop(now + 3.2);
}

function scheduleNextThunder() {
  // 14–32 s between cracks. Long-tail randomness keeps it from feeling
  // metronomic.
  const delay = 14000 + Math.random() * 18000;
  window.setTimeout(() => {
    playThunder();
    scheduleNextThunder();
  }, delay);
}

export function startAudio() {
  if (started || typeof window === "undefined") return;
  started = true;

  const stored = loadStoredMute();
  if (stored !== null) {
    muted = stored;
  } else {
    muted = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  startRain();
  scheduleNextThunder();

  // Notify any UI listeners of the resolved initial state.
  window.dispatchEvent(new CustomEvent("audio:state", { detail: { muted } }));
}

export function setMuted(value: boolean) {
  muted = value;
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, value ? "1" : "0");
    window.dispatchEvent(new CustomEvent("audio:state", { detail: { muted } }));
  }
  if (rainNode && ctx) {
    rainNode.gain.gain.setTargetAtTime(
      value ? 0 : RAIN_GAIN,
      ctx.currentTime,
      0.05,
    );
  }
}

export function isMuted(): boolean {
  return muted;
}

export function isStarted(): boolean {
  return started;
}

export function getStoredMute(): boolean | null {
  return loadStoredMute();
}
