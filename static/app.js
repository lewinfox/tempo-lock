/* tempo-lock front end. Vanilla JS, no build step. */
(() => {
  const $ = (id) => document.getElementById(id);
  const fmtTime = (t) => {
    if (!isFinite(t)) return "0:00.000";
    const m = Math.floor(t / 60), s = t - m * 60;
    return `${m}:${s.toFixed(3).padStart(6, "0")}`;
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  // ID3 text is attacker-controlled - anyone can craft an MP3 - so escape before innerHTML
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ESC[c]);
  const stem = (name) => String(name).replace(/\.[^.\/]+$/, "");
  // Below this many samples per pixel a min/max bar is under a pixel tall, so drawWave
  // switches from the envelope to a line through the samples themselves.
  const ENVELOPE_MIN_SPP = 8;
  const MIN_SPAN = 0.002; // tightest zoom, seconds: ~90 samples across the canvas
  const bsearch = (arr, x) => { // first index with arr[i] >= x
    let lo = 0, hi = arr.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
    return lo;
  };

  // ------------------------------------------------------------------ state
  const state = {
    track: null,            // server JSON
    tracks: { a: null, b: null }, // {buffer, pyramid, duration, beats, isDown, peaks}
    view: [0, 1],           // visible window on the A (original) timeline, seconds
    src: "a",               // which version is playing / shown as active
    ctx: null, node: null, gain: null,
    playing: false, startCtx: 0, startOffset: 0, pausedAt: 0,
    clickTimer: null, nextBeat: 0,
    raf: 0,
    renderSeq: 0,           // bumped per render so a stale follow-up poll gives up
    openSeq: 0,             // bumped per job opened so a stale load gives up
  };

  // --------------------------------------------------------- time mapping
  // piecewise-linear map between the original (A) and straightened (B) timelines
  function mapTime(t, from) {
    const g = state.track && state.track.rendered && state.track.rendered.grid;
    if (!g) return t;
    const src = from === "a" ? g.source_beats : g.target_beats;
    const dst = from === "a" ? g.target_beats : g.source_beats;
    if (t <= src[0]) return t - src[0] + dst[0];
    const n = src.length;
    if (t >= src[n - 1]) return t - src[n - 1] + dst[n - 1];
    const i = bsearch(src, t);
    const s0 = src[i - 1], s1 = src[i], d0 = dst[i - 1], d1 = dst[i];
    return d0 + (t - s0) / (s1 - s0) * (d1 - d0);
  }
  const toA = (t, from) => (from === "a" ? t : mapTime(t, "b"));
  const fromA = (t, to) => (to === "a" ? t : mapTime(t, "a"));

  // ------------------------------------------------------------- audio
  function ensureCtx() {
    if (!state.ctx) {
      state.ctx = new (window.AudioContext || window.webkitAudioContext)();
      state.gain = state.ctx.createGain();
      state.gain.connect(state.ctx.destination);
    }
    if (state.ctx.state === "suspended") state.ctx.resume();
    return state.ctx;
  }

  async function loadAudio(which, url) {
    const ctx = ensureCtx();
    const t = state.tracks[which]; // opening another job replaces this object, so a late result lands nowhere
    const res = await fetch(url);
    if (!res.ok) throw new Error(`audio ${res.status}`);
    const buf = await ctx.decodeAudioData(await res.arrayBuffer());
    t.buffer = buf;
    t.duration = buf.duration;
    t.pyramid = buildPyramid(buf, 256);
    draw();
  }

  // min/max per 256-sample block, used when zoomed out; raw samples when zoomed in
  function buildPyramid(buf, block) {
    const n = buf.length, nb = Math.ceil(n / block);
    const mins = new Float32Array(nb), maxs = new Float32Array(nb);
    const chans = [];
    for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
    for (let b = 0; b < nb; b++) {
      let mn = 1, mx = -1;
      const end = Math.min(n, (b + 1) * block);
      for (let i = b * block; i < end; i++) {
        let v = 0;
        for (let c = 0; c < chans.length; c++) v += chans[c][i];
        v /= chans.length;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      mins[b] = mn; maxs[b] = mx;
    }
    return { block, mins, maxs, chans, n, sr: buf.sampleRate };
  }

  function position() { // playhead in the active timeline
    if (!state.playing) return state.pausedAt;
    return state.startOffset + (state.ctx.currentTime - state.startCtx);
  }

  function play(offset) {
    const t = state.tracks[state.src];
    if (!t || !t.buffer) return;
    const ctx = ensureCtx();
    stopNode();
    if (offset === undefined) offset = state.pausedAt;
    offset = clamp(offset, 0, t.duration - 0.01);
    const node = ctx.createBufferSource();
    node.buffer = t.buffer;
    node.connect(state.gain);
    node.onended = () => { if (state.node === node) { state.playing = false; state.pausedAt = t.duration; stopClicks(); $("play").textContent = "▶"; draw(); } };
    node.start(0, offset);
    state.node = node;
    state.startCtx = ctx.currentTime;
    state.startOffset = offset;
    state.playing = true;
    $("play").textContent = "❚❚";
    startClicks();
    loop();
  }
  function stopNode() {
    if (state.node) { try { state.node.onended = null; state.node.stop(); } catch (e) {} state.node = null; }
  }
  function pause() {
    if (!state.playing) return;
    state.pausedAt = position();
    stopNode();
    state.playing = false;
    stopClicks();
    $("play").textContent = "▶";
    draw();
  }
  function seek(tActive) {
    const t = state.tracks[state.src];
    tActive = clamp(tActive, 0, t ? t.duration : 0);
    if (state.playing) play(tActive); else { state.pausedAt = tActive; draw(); }
  }
  function switchSource(which) {
    if (which === state.src) return;
    if (!state.tracks[which] || !state.tracks[which].buffer) return;
    const pos = position();
    const mapped = fromA(toA(pos, state.src), which);
    state.src = which;
    $("src-a").classList.toggle("on", which === "a");
    $("src-b").classList.toggle("on", which === "b");
    if (state.playing) play(mapped); else { state.pausedAt = mapped; draw(); }
  }

  // click track: schedule short blips on the beat grid of the active version
  function scheduleClick(when, down) {
    const ctx = state.ctx;
    const osc = ctx.createOscillator(), g = ctx.createGain();
    osc.frequency.value = down ? 1760 : 1175;
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(0.5, when + 0.001);
    g.gain.exponentialRampToValueAtTime(0.0001, when + 0.03);
    osc.connect(g); g.connect(ctx.destination);
    osc.start(when); osc.stop(when + 0.04);
  }
  function startClicks() {
    stopClicks();
    if (!$("click").checked) return;
    const t = state.tracks[state.src];
    if (!t || !t.beats) return;
    state.nextBeat = bsearch(t.beats, state.startOffset);
    const tick = () => {
      const ctx = state.ctx, horizon = position() + 0.2;
      while (state.nextBeat < t.beats.length && t.beats[state.nextBeat] < horizon) {
        const tb = t.beats[state.nextBeat];
        if (tb >= state.startOffset) scheduleClick(state.startCtx + (tb - state.startOffset), t.isDown[state.nextBeat]);
        state.nextBeat++;
      }
    };
    tick();
    state.clickTimer = setInterval(tick, 50);
  }
  function stopClicks() { if (state.clickTimer) { clearInterval(state.clickTimer); state.clickTimer = null; } }

  // ------------------------------------------------------------ drawing
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();

  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { g, w, h };
  }

  function drawWave(which) {
    const cv = $(which === "a" ? "wave-a" : "wave-b");
    const t = state.tracks[which];
    const { g, w, h } = setupCanvas(cv);
    g.clearRect(0, 0, w, h);
    if (!t) return;
    const v0 = fromA(state.view[0], which), v1 = fromA(state.view[1], which);
    const span = v1 - v0, mid = h / 2;
    const wave = color(which === "a" ? "--wave-a" : "--wave-b");

    // waveform
    g.fillStyle = wave;
    if (t.pyramid) {
      const p = t.pyramid, sr = p.sr, spp = span * sr / w; // samples per pixel
      const sampleAt = (i) => { let v = 0; for (let c = 0; c < p.chans.length; c++) v += p.chans[c][i]; return v / p.chans.length; };
      const yOf = (v) => mid - v * mid * 0.95;
      if (spp < ENVELOPE_MIN_SPP) {
        // Zoomed past a few samples per column a min/max bar has no height left, so the
        // envelope degenerates into a 1px dotted line. Trace the samples as a connected
        // curve instead, the way a DAW does, and mark each one once it owns a few pixels.
        const i0 = Math.max(0, Math.floor(v0 * sr)), i1 = Math.min(p.n - 1, Math.ceil(v1 * sr));
        g.strokeStyle = wave; g.lineWidth = 1; g.lineJoin = "round"; g.beginPath();
        for (let i = i0; i <= i1; i++) {
          const x = (i / sr - v0) / span * w;
          if (i === i0) g.moveTo(x, yOf(sampleAt(i))); else g.lineTo(x, yOf(sampleAt(i)));
        }
        g.stroke();
        if (spp < 0.34) for (let i = i0; i <= i1; i++) g.fillRect((i / sr - v0) / span * w - 1.5, yOf(sampleAt(i)) - 1.5, 3, 3);
      } else for (let x = 0; x < w; x++) {
        const s0 = Math.floor((v0 + x / w * span) * sr), s1 = Math.floor((v0 + (x + 1) / w * span) * sr);
        if (s1 < 0 || s0 >= p.n) continue;
        let mn = 1, mx = -1;
        if (spp > p.block * 2) {
          for (let b = Math.max(0, Math.floor(s0 / p.block)); b < Math.min(p.mins.length, Math.ceil(s1 / p.block)); b++) {
            if (p.mins[b] < mn) mn = p.mins[b]; if (p.maxs[b] > mx) mx = p.maxs[b];
          }
        } else {
          for (let i = Math.max(0, s0); i < Math.min(p.n, Math.max(s0 + 1, s1)); i++) {
            const v = sampleAt(i);
            if (v < mn) mn = v; if (v > mx) mx = v;
          }
        }
        if (mn > mx) continue;
        const y0 = yOf(mx), y1 = yOf(mn);
        g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
      }
    } else if (t.peaks) { // server-side peaks before audio has decoded
      const n = t.peaks.min.length;
      for (let x = 0; x < w; x++) {
        const b0 = Math.floor((v0 + x / w * span) / t.duration * n), b1 = Math.max(b0 + 1, Math.floor((v0 + (x + 1) / w * span) / t.duration * n));
        let mn = 127, mx = -127;
        for (let b = Math.max(0, b0); b < Math.min(n, b1); b++) { if (t.peaks.min[b] < mn) mn = t.peaks.min[b]; if (t.peaks.max[b] > mx) mx = t.peaks.max[b]; }
        if (mn > mx) continue;
        g.fillRect(x, mid - mx / 127 * mid * 0.95, 1, Math.max(1, (mx - mn) / 127 * mid * 0.95));
      }
    }

    // beat grid
    if (t.beats) {
      const i0 = bsearch(t.beats, v0), i1 = bsearch(t.beats, v1);
      const showNumbers = (i1 - i0) < w / 28;
      g.font = "11px system-ui, sans-serif"; g.textAlign = "left";
      for (let i = i0; i < i1; i++) {
        const x = (t.beats[i] - v0) / span * w;
        const down = t.isDown[i];
        g.fillStyle = down ? color("--down") : color("--beat");
        g.fillRect(Math.round(x) - (down ? 1 : 0), 0, down ? 2 : 1, h);
        if (showNumbers && down) {
          g.fillStyle = color("--down");
          g.fillText(String(t.barOf ? t.barOf[i] : ""), x + 3, 12);
        }
      }
      // dropped (spurious) detections, faint red
      if (t.dropped) {
        g.fillStyle = "rgba(255,90,90,.5)";
        for (const d of t.dropped) if (d >= v0 && d <= v1) g.fillRect((d - v0) / span * w, h - 8, 1, 8);
      }
    }

    // time ruler
    g.fillStyle = "rgba(255,255,255,.35)"; g.font = "10px system-ui, sans-serif"; g.textAlign = "left";
    const step = niceStep(span / (w / 90));
    for (let tt = Math.ceil(v0 / step) * step; tt < v1; tt += step) {
      const x = (tt - v0) / span * w;
      g.fillRect(x, h - 14, 1, 4);
      g.fillText(fmtTime(tt).replace(/\.\d+$/, step < 1 ? (m) => m.slice(0, 3) : ""), x + 2, h - 3);
    }

    // playhead
    const pos = fromA(toA(position(), state.src), which);
    if (pos >= v0 && pos <= v1) {
      g.fillStyle = color("--play");
      g.globalAlpha = state.src === which ? 1 : 0.45;
      g.fillRect((pos - v0) / span * w - 0.5, 0, 1.5, h);
      g.globalAlpha = 1;
    }
    drawOverview(which, v0, v1);
  }

  function niceStep(raw) {
    const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    for (const s of steps) if (s >= raw) return s;
    return 600;
  }

  function drawOverview(which, v0, v1) {
    const box = $(which === "a" ? "overview-a" : "overview-b");
    const t = state.tracks[which];
    if (!t) return;
    let cv = box.querySelector("canvas"), win = box.querySelector(".win");
    if (!cv) {
      cv = document.createElement("canvas"); box.appendChild(cv);
      win = document.createElement("div"); win.className = "win"; box.appendChild(win);
      box.addEventListener("mousedown", (e) => {
        const move = (ev) => {
          const r = box.getBoundingClientRect();
          const centre = fromA(toA(clamp((ev.clientX - r.left) / r.width, 0, 1) * t.duration, which), "a");
          const span = state.view[1] - state.view[0];
          setView(centre - span / 2, centre + span / 2);
        };
        move(e);
        const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
        window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
      });
    }
    if (t.pyramid || t.peaks) {
      const { g, w, h } = setupCanvas(cv);
      g.clearRect(0, 0, w, h);
      g.fillStyle = "rgba(255,255,255,.35)";
      const src = t.pyramid ? t.pyramid : null;
      const n = src ? src.mins.length : t.peaks.min.length;
      for (let x = 0; x < w; x++) {
        const b0 = Math.floor(x / w * n), b1 = Math.max(b0 + 1, Math.floor((x + 1) / w * n));
        let mn = 1, mx = -1;
        for (let b = b0; b < Math.min(n, b1); b++) {
          const a = src ? src.mins[b] : t.peaks.min[b] / 127, c = src ? src.maxs[b] : t.peaks.max[b] / 127;
          if (a < mn) mn = a; if (c > mx) mx = c;
        }
        g.fillRect(x, h / 2 - mx * h / 2, 1, Math.max(1, (mx - mn) * h / 2));
      }
    }
    win.style.left = `${clamp(v0 / t.duration, 0, 1) * 100}%`;
    win.style.width = `${clamp((v1 - v0) / t.duration, 0.002, 1) * 100}%`;
  }

  const SECTION_COLORS = ["rgba(90,209,255,.16)", "rgba(125,255,167,.16)", "rgba(255,93,143,.16)", "rgba(255,180,84,.16)", "rgba(200,140,255,.16)"];
  const SECTION_SOLID = ["#5ad1ff", "#7dffa7", "#ff5d8f", "#ffb454", "#c88cff"];

  // The charts for the original and the straightened track share their BPM scale, so a
  // flatter line or a narrower histogram on the second really means a steadier tempo.
  function analysisOf(which) {
    const t = state.track;
    if (!t) return null;
    const a = which === "a" ? t.analysis : t.rendered && t.rendered.analysis;
    return a && !a.error && a.bpm_curve && a.bpm_curve.length ? a : null;
  }
  function targetOf(which) { // target tempo at the detected beat level
    if (which === "b") return state.track.rendered.bpm / (state.track.rendered.level || 1);
    const a = state.track.analysis;
    return (parseFloat($("bpm").value) || a.suggested_bpm) / (parseFloat($("level").value) || 1);
  }
  function charted() { return ["a", "b"].filter((w) => analysisOf(w)).map((w) => [analysisOf(w), targetOf(w)]); }

  function bpmRange() {
    let lo = Infinity, hi = -Infinity;
    for (const [a, target] of charted()) {
      lo = Math.min(lo, a.min_bpm, target); hi = Math.max(hi, a.max_bpm, target);
      for (const sec of a.sections || []) { lo = Math.min(lo, sec.bpm); hi = Math.max(hi, sec.bpm); }
    }
    const pad = Math.max(1, (hi - lo) * 0.25);
    return [lo - pad, hi + pad];
  }

  function histRange() {
    let lo = Infinity, hi = -Infinity;
    for (const [a, target] of charted()) {
      for (const v of a.bpm_curve) { if (v < lo) lo = v; if (v > hi) hi = v; }
      lo = Math.min(lo, target); hi = Math.max(hi, target);
    }
    // bin width: a "nice" number giving ~30 bins across the observed range
    const bw = niceStep(Math.max(0.1, (hi - lo) / 30));
    return { lo, hi, bw, b0: Math.floor(lo / bw) - 1, b1: Math.ceil(hi / bw) + 1 };
  }

  function drawHist(which) {
    const cv = $(which === "a" ? "hist" : "hist-b");
    const { g, w, h } = setupCanvas(cv);
    g.clearRect(0, 0, w, h);
    const a = analysisOf(which);
    if (!a) return;
    const target = targetOf(which);
    const vals = a.bpm_curve;
    const { lo, hi, bw, b0, b1 } = histRange(), nb = b1 - b0;
    const sections = (a.sections && a.sections.length > 1) ? a.sections : [{ start_beat: 0, end_beat: vals.length }];
    const counts = sections.map(() => new Float64Array(nb));
    sections.forEach((sec, si) => {
      for (let i = sec.start_beat; i < Math.min(sec.end_beat, vals.length); i++) counts[si][Math.floor(vals[i] / bw) - b0]++;
    });
    const totals = new Float64Array(nb);
    counts.forEach((c) => { for (let i = 0; i < nb; i++) totals[i] += c[i]; });
    let peak = 1; for (const t of totals) if (t > peak) peak = t;
    const padL = 6, padB = 16, padT = sections.length > 1 ? 16 : 6, plotH = h - padB - padT, colW = (w - padL * 2) / nb;
    const X = (bpm) => padL + (bpm / bw - b0) * colW;
    // bars, stacked by section
    for (let i = 0; i < nb; i++) {
      let y = h - padB;
      sections.forEach((_, si) => {
        const c = counts[si][i]; if (!c) return;
        const bh = c / peak * plotH;
        g.fillStyle = sections.length > 1 ? SECTION_SOLID[si % SECTION_SOLID.length] : color(which === "a" ? "--wave-a" : "--wave-b");
        g.globalAlpha = 0.85;
        g.fillRect(padL + i * colW + 0.5, y - bh, Math.max(1, colW - 1), bh);
        g.globalAlpha = 1;
        y -= bh;
      });
    }
    // axis labels
    g.fillStyle = "rgba(255,255,255,.4)"; g.font = "10px system-ui, sans-serif"; g.textAlign = "center";
    const labStep = niceStep((hi - lo) / 4);
    for (let b = Math.ceil((b0 * bw) / labStep) * labStep; b <= b1 * bw; b += labStep) {
      const x = X(b); if (x < padL || x > w - padL) continue;
      g.fillRect(x, h - padB, 1, 3);
      g.fillText(b.toFixed(labStep < 1 ? 1 : 0), x, h - 3);
    }
    // target
    g.setLineDash([4, 4]); g.strokeStyle = color("--accent"); g.beginPath(); g.moveTo(X(target), 4); g.lineTo(X(target), h - padB); g.stroke(); g.setLineDash([]);
    // section medians
    if (sections.length > 1) sections.forEach((sec, si) => {
      g.fillStyle = SECTION_SOLID[si % SECTION_SOLID.length]; g.textAlign = "center";
      g.fillText(sec.bpm.toFixed(1), clamp(X(sec.bpm), 16, w - 16), 10);
    });
    g.textAlign = "left";
  }

  function drawTempo(which) {
    const cv = $(which === "a" ? "tempo" : "tempo-b");
    const { g, w, h } = setupCanvas(cv);
    g.clearRect(0, 0, w, h);
    const a = analysisOf(which);
    if (!a) return;
    const level = which === "a" ? parseFloat($("level").value) || 1 : state.track.rendered.level || 1;
    const target = targetOf(which);
    const dur = a.duration;
    const [lo, hi] = bpmRange();
    const X = (t) => t / dur * w, Y = (b) => h - (b - lo) / (hi - lo) * h;
    // steady-tempo sections (only interesting when there is more than one)
    if (a.sections && a.sections.length > 1) {
      a.sections.forEach((sec, i) => {
        g.fillStyle = SECTION_COLORS[i % SECTION_COLORS.length];
        g.fillRect(X(sec.start), 0, X(sec.end) - X(sec.start), h);
        g.strokeStyle = SECTION_SOLID[i % SECTION_SOLID.length]; g.lineWidth = 1; g.globalAlpha = 0.7;
        g.beginPath(); g.moveTo(X(sec.start), Y(sec.bpm)); g.lineTo(X(sec.end), Y(sec.bpm)); g.stroke(); g.globalAlpha = 1;
      });
    }
    // gridlines
    g.strokeStyle = "rgba(255,255,255,.08)"; g.fillStyle = "rgba(255,255,255,.4)"; g.font = "10px system-ui, sans-serif";
    const step = niceStep((hi - lo) / 4);
    for (let b = Math.ceil(lo / step) * step; b < hi; b += step) {
      g.beginPath(); g.moveTo(0, Y(b)); g.lineTo(w, Y(b)); g.stroke();
      g.fillText(b.toFixed(step < 1 ? 1 : 0), 4, Math.max(10, Y(b) - 2));
    }
    // visible window shading (the view is kept in A time)
    const v0 = fromA(state.view[0], which), v1 = fromA(state.view[1], which);
    g.fillStyle = "rgba(90,209,255,.07)";
    g.fillRect(X(v0), 0, X(v1) - X(v0), h);
    // curve
    g.strokeStyle = color(which === "a" ? "--wave-a" : "--wave-b"); g.lineWidth = 1.5; g.beginPath();
    for (let i = 0; i < a.bpm_curve.length; i++) {
      const x = X(a.bpm_times[i]), y = Y(clamp(a.bpm_curve[i], lo, hi));
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
    // target
    g.setLineDash([4, 4]); g.strokeStyle = color("--accent"); g.beginPath(); g.moveTo(0, Y(target)); g.lineTo(w, Y(target)); g.stroke(); g.setLineDash([]);
    g.fillStyle = color("--accent"); g.textAlign = "right"; g.fillText(`${target.toFixed(2)} BPM${level !== 1 ? ` (detected level, ×${level})` : ""}`, w - 4, Y(target) - 3); g.textAlign = "left";
    // deliberate tempo changes
    g.font = "13px system-ui, sans-serif"; g.textAlign = "center";
    for (const c of a.tempo_changes || []) {
      g.setLineDash([2, 3]); g.strokeStyle = color("--accent"); g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(X(c.time), 16); g.lineTo(X(c.time), h); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
      g.fillStyle = color("--accent"); g.fillText("⚠", X(c.time), 13);
    }
    g.textAlign = "left";
    // playhead
    const pos = fromA(toA(position(), state.src), which);
    g.fillStyle = color("--play"); g.fillRect(X(pos) - 0.5, 0, 1.5, h);
  }

  function draw() {
    drawTempo("a");
    drawHist("a");
    drawWave("a");
    if (state.tracks.b) drawWave("b");
    if (analysisOf("b")) { drawTempo("b"); drawHist("b"); }
    $("time").textContent = fmtTime(position());
  }

  function loop() {
    cancelAnimationFrame(state.raf);
    const step = () => {
      if (!state.playing) { draw(); return; }
      if ($("follow").checked) {
        const posA = toA(position(), state.src);
        const span = state.view[1] - state.view[0];
        if (posA > state.view[1] || posA < state.view[0]) setView(posA - span * 0.1, posA + span * 0.9, true);
      }
      draw();
      state.raf = requestAnimationFrame(step);
    };
    state.raf = requestAnimationFrame(step);
  }

  function setView(v0, v1, noDraw) {
    const dur = state.tracks.a ? state.tracks.a.duration : 1;
    let span = clamp(v1 - v0, Math.min(MIN_SPAN, dur), dur);
    v0 = clamp(v0, 0, Math.max(0, dur - span)); v1 = v0 + span;
    state.view = [v0, v1];
    if (!noDraw) draw();
  }

  function wireWave(which) {
    const cv = $(which === "a" ? "wave-a" : "wave-b");
    let downX = null, moved = false, startView = null;
    const xToTime = (clientX) => {
      const r = cv.getBoundingClientRect();
      const v0 = fromA(state.view[0], which), v1 = fromA(state.view[1], which);
      return v0 + (clientX - r.left) / r.width * (v1 - v0);
    };
    cv.addEventListener("mousedown", (e) => { downX = e.clientX; moved = false; startView = [...state.view]; e.preventDefault(); });
    window.addEventListener("mousemove", (e) => {
      if (downX === null) return;
      const dx = e.clientX - downX;
      if (Math.abs(dx) > 3) moved = true;
      if (moved) {
        const r = cv.getBoundingClientRect();
        const span = startView[1] - startView[0];
        setView(startView[0] - dx / r.width * span, startView[1] - dx / r.width * span);
      }
    });
    window.addEventListener("mouseup", (e) => {
      if (downX === null) return;
      if (!moved) {
        const t = xToTime(e.clientX);
        if (which !== state.src) switchSource(which);
        seek(t);
      }
      downX = null;
    });
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const frac = (e.clientX - r.left) / r.width;
      const span = state.view[1] - state.view[0];
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const d = (e.deltaX || e.deltaY) / r.width * span;
        setView(state.view[0] + d, state.view[1] + d);
      } else {
        const factor = Math.exp(e.deltaY * 0.0015);
        const anchor = state.view[0] + frac * span;
        const ns = clamp(span * factor, MIN_SPAN, state.tracks.a.duration);
        setView(anchor - frac * ns, anchor + (1 - frac) * ns);
      }
    }, { passive: false });
  }

  for (const which of ["a", "b"]) {
    const cv = $(which === "a" ? "tempo" : "tempo-b");
    cv.addEventListener("click", (e) => {
      const a = analysisOf(which); if (!a) return;
      const r = cv.getBoundingClientRect();
      const t = (e.clientX - r.left) / r.width * a.duration;
      seek(fromA(toA(t, which), state.src));
    });
  }

  // ---------------------------------------------------------------- flow
  // "loading" is the only state worth animating; the rest are just labels.
  const MODEL_LABEL = { cold: "not loaded", loading: "loading…", ready: "ready", failed: "failed to load", unavailable: "" };

  function showHealth(h) {
    const detector = h.beat_this ? "Beat This!" : "librosa fallback";
    const model = h.model === "loading" ? '<span class="spinner"></span> loading…' : MODEL_LABEL[h.model] ?? h.model;
    $("health").innerHTML = [
      `detector: <b>${detector}</b>${model ? ` <span class="${h.model === "failed" ? "bad" : ""}">${model}</span>` : ""}`,
      `stretcher: <b class="${h.rubberband ? "" : "bad"}">${h.rubberband || "rubberband missing - rendering disabled"}</b>`,
      `encoder: <b>${h.ffmpeg ? "ffmpeg" : "libsndfile"}</b>`,
    ].join(" · ");
  }

  // The server pre-loads the model at boot, but ask anyway: warm-up may be switched off,
  // and the point is that the ~7 s wait happens while the user is still choosing a file.
  async function health() {
    try {
      let h = await (await fetch("/api/health")).json();
      showHealth(h);
      if (h.model === "cold" || h.model === "failed") {
        await fetch("/api/warm", { method: "POST" }).catch(() => {});
      } else if (h.model !== "loading") return;
      for (let i = 0; i < 120; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        h = await (await fetch("/api/health")).json();
        showHealth(h);
        if (h.model !== "loading") return;
      }
    } catch (e) { $("health").textContent = "server unreachable"; }
  }

  // Any long job raises the bar at the top of the window; jobs can overlap (the audio
  // fetch outlives the analysis poll), so count them rather than flipping a flag.
  let busyJobs = 0;
  function busy(on) {
    busyJobs = Math.max(0, busyJobs + (on ? 1 : -1));
    $("busy").hidden = busyJobs === 0;
  }

  function status(msg, spin) {
    const el = $("status");
    el.hidden = !msg;
    el.innerHTML = msg ? `${spin ? '<span class="spinner"></span>' : ""}<span>${msg}</span>` : "";
  }

  function working(el, msg) {
    el.classList.add("working");
    el.innerHTML = `<span class="spinner"></span> ${msg}`;
  }
  function settled(el, msg) {
    el.classList.remove("working");
    el.textContent = msg;
  }

  async function upload(file) {
    busy(true);
    try {
      await runUpload(file);
    } finally {
      busy(false);
    }
  }

  // Clear the page for another job.
  function resetView() {
    state.openSeq++; state.renderSeq++;
    $("result").hidden = true; $("rendered-card").hidden = true;
    pause(); state.tracks = { a: null, b: null }; state.track = null; state.src = "a"; state.pausedAt = 0;
    $("src-a").classList.add("on"); $("src-b").classList.remove("on"); $("src-b").disabled = true; $("download").hidden = true;
    $("level").value = "1"; settled($("render-status"), "");
    $("redetect").hidden = true;
    markOpenJob();
  }

  async function runUpload(file) {
    resetView();
    status(`Uploading ${file.name}…`, true);
    const fd = new FormData(); fd.append("file", file);
    const res = await fetch(`/api/tracks?detector=${$("detector").value}`, { method: "POST", body: fd });
    if (!res.ok) { status(`Upload failed: ${await res.text()}`); return; }
    const { id } = await res.json();
    refreshJobs();
    await runOpen(id);
  }

  async function openJob(id) {
    busy(true);
    try {
      await runOpen(id);
    } finally {
      busy(false);
    }
  }

  // Load a job into the page: its audio, the detected beats and, if it has one, the
  // straightened version with the settings it was made with.
  async function runOpen(id) {
    resetView();
    const seq = state.openSeq;
    const stale = () => seq !== state.openSeq;
    history.replaceState(null, "", `?job=${id}`);
    state.track = { id };
    markOpenJob();
    status("Opening job…", true);
    let res = await fetch(`/api/tracks/${id}`);
    if (stale()) return;
    if (!res.ok) { status("That job is no longer on the server."); history.replaceState(null, "", location.pathname); refreshJobs(); return; }
    let track = await res.json();
    state.tracks.a = { duration: 1, beats: null, isDown: null };
    const audioP = loadAudio("a", `/api/tracks/${id}/audio/original`).catch(() => {});
    if (track.status === "analysing") {
      status("Detecting beats…", true);
      const t0 = performance.now();
      track = await poll(id, (t) => t.status !== "analysing");
      if (stale()) return;
      if (track.analysis) status(`Analysed in ${((performance.now() - t0) / 1000).toFixed(1)} s with ${track.analysis.detector}.`);
      refreshJobs();
    } else status("");
    $("redetect").hidden = false;
    if (!track.analysis) { status(`Beat detection failed: ${track.error}. Pick a detector above and click “Detect beats again”.`); return; }
    state.track = track;
    applyAnalysis(track);
    $("result").hidden = false;
    setView(0, Math.min(track.analysis.duration, 30));
    const params = track.rendered && track.rendered.params;
    if (params) {
      $("bpm").value = params.target_bpm; $("level").value = String(params.level); $("engine").value = params.engine;
    }
    if (track.status === "rendering") await renderTrack(true);
    else if (track.rendered) await showRendered(track);
    else if (track.status === "error") settled($("render-status"), `Last straighten failed: ${track.error}`);
    await audioP;
    draw();
  }

  async function redetect() {
    const id = state.track && state.track.id; if (!id) return;
    if (state.track.rendered && !confirm("Detecting the beats again deletes this job's straightened file. Carry on?")) return;
    const res = await fetch(`/api/tracks/${id}/analyse?detector=${$("detector").value}`, { method: "POST" });
    if (!res.ok) { let msg = await res.text(); try { msg = JSON.parse(msg).detail || msg; } catch (e) {} status(msg); return; }
    refreshJobs();
    await openJob(id);
  }

  async function poll(id, done) {
    for (;;) {
      const t = await (await fetch(`/api/tracks/${id}`)).json();
      if (done(t)) return t;
      await new Promise((r) => setTimeout(r, 700));
    }
  }

  function barNumbers(isDown) {
    const out = new Array(isDown.length).fill(""); let bar = 0;
    for (let i = 0; i < isDown.length; i++) if (isDown[i]) out[i] = String(++bar);
    return out;
  }

  function renderWarning(a) {
    const box = $("warn");
    const changes = a.tempo_changes || [];
    box.hidden = !changes.length;
    if (!changes.length) { box.innerHTML = ""; return; }
    const items = changes.map((c, i) => {
      const pct = ((c.ratio - 1) * 100).toFixed(0);
      const what = c.kind === "tempo change" ? `${c.from_bpm.toFixed(1)} → ${c.to_bpm.toFixed(1)} BPM (${pct > 0 ? "+" : ""}${pct}%)` : `${c.kind} feel, ${c.from_bpm.toFixed(1)} → ${c.to_bpm.toFixed(1)} BPM`;
      return `<div><b>${fmtTime(c.time)}</b> ${what}<a data-seek="${c.time}">jump to</a></div>`;
    }).join("");
    const secs = a.sections.map((sec) => `${sec.bpm.toFixed(1)}`).join(" / ");
    box.innerHTML = `<div class="icon">⚠</div><div>
      <div><b>${changes.length === 1 ? "A tempo change that looks deliberate" : `${changes.length} tempo changes that look deliberate`}</b>:
      the tempo steps to a new value and stays there, rather than drifting. Sections: ${secs} BPM.</div>${items}
      <div class="hint" style="margin:6px 0 0">Straightening to one BPM will speed up or slow down whole sections, not just tidy the drift. Check the histogram: separate humps are separate tempos.</div></div>`;
    box.querySelectorAll("a[data-seek]").forEach((el) => el.addEventListener("click", () => {
      const tA = parseFloat(el.dataset.seek);
      const span = state.view[1] - state.view[0];
      setView(tA - span * 0.3, tA + span * 0.7);
      seek(fromA(tA, state.src));
    }));
  }

  function showTrackInfo(track) {
    const info = track.info || {}, tags = info.tags || {};
    const titled = Boolean(tags.title);
    const sub = [tags.artist, tags.album, (tags.date || "").slice(0, 4)].filter(Boolean);
    const rate = info.sample_rate ? info.sample_rate / 1000 : null;
    const tech = [
      info.codec && info.codec.toUpperCase().replace(/^PCM_.*/, "PCM"),
      info.bitrate_kbps && `${info.bitrate_kbps} kbps${info.lossless ? "" : " (source)"}`,
      rate && `${rate % 1 ? rate.toFixed(1) : rate} kHz`,
      info.channels && (info.channels === 1 ? "mono" : info.channels === 2 ? "stereo" : `${info.channels} ch`),
      info.lossless && "lossless",
    ].filter(Boolean);
    $("trackinfo").innerHTML =
      `<div class="t">${esc(titled ? tags.title : stem(track.name))}</div>` +
      (sub.length ? `<div class="a">${sub.map(esc).join(" · ")}</div>` : "") +
      (titled ? "" : `<div class="tech untagged">no title tag - showing the filename</div>`) +
      (tech.length ? `<div class="tech">${tech.map(esc).join(" · ")}</div>` : "");
  }

  const statHtml = ([k, v, s, cls]) => `<div class="stat ${cls || ""}"><div class="k">${k}</div><div class="v">${v} <small>${s}</small></div></div>`;

  function applyAnalysis(track) {
    const a = track.analysis;
    const changes = a.tempo_changes || [];
    const t = state.tracks.a || {};
    Object.assign(t, { duration: t.buffer ? t.buffer.duration : a.duration, beats: a.beats, isDown: a.is_downbeat, barOf: barNumbers(a.is_downbeat), peaks: track.peaks, dropped: a.dropped_beats });
    state.tracks.a = t;
    const spread = a.max_bpm - a.min_bpm;
    $("stats").innerHTML = [
      ["Detector", a.detector === "beat_this" ? "Beat This!" : a.detector, ""],
      ["Beats", a.beats.length, `${a.is_downbeat.filter(Boolean).length} downbeats`],
      ["Median tempo", a.median_bpm.toFixed(2), "BPM"],
      ["Tempo range", `${a.min_bpm.toFixed(1)}–${a.max_bpm.toFixed(1)}`, "BPM, 2nd–98th pct"],
      ["Drift", `±${(spread / 2 / a.median_bpm * 100).toFixed(1)}%`, `${spread.toFixed(1)} BPM spread`],
      ["Tempo changes", changes.length ? `⚠ ${changes.length}` : "none", changes.length ? "look deliberate" : "drift only", changes.length ? "flag" : ""],
      ["Length", fmtTime(a.duration), ""],
    ].map(statHtml).join("");
    showTrackInfo(track);
    renderWarning(a);
    $("bpm").value = a.suggested_bpm.toFixed(2).replace(/\.00$/, "");
    $("render").disabled = false;
  }

  // `resume` picks up a render that was already running when the job was opened.
  async function renderTrack(resume) {
    const btn = $("render"), st = $("render-status");
    const bpm = parseFloat($("bpm").value);
    if (!resume && !(bpm >= 20 && bpm <= 400)) { settled(st, "BPM must be between 20 and 400"); return; }
    // Rubber Band R3 on a long track is tens of seconds of silence otherwise: put the
    // work on the button that started it, in the status line, and on the top bar.
    btn.disabled = true; btn.classList.add("busy");
    btn.innerHTML = '<span class="spinner on-accent"></span> Straightening…';
    working(st, "stretching with Rubber Band…");
    busy(true);
    try {
      await runRender(bpm, st, resume);
    } finally {
      busy(false);
      btn.disabled = false; btn.classList.remove("busy"); btn.textContent = "Straighten";
    }
  }

  async function runRender(bpm, st, resume) {
    const t0 = performance.now();
    const id = state.track.id, openSeq = state.openSeq;
    if (!resume) {
      const res = await fetch(`/api/tracks/${id}/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target_bpm: bpm, engine: $("engine").value, level: parseFloat($("level").value) }) });
      if (!res.ok) {
        let msg = await res.text();
        try { msg = JSON.parse(msg).detail || msg; } catch (e) {}
        settled(st, msg); return;
      }
      refreshJobs();
    }
    state.renderSeq++;
    $("charts-b").hidden = true; $("stats-b").innerHTML = "";
    const track = await poll(id, (t) => t.status !== "rendering");
    if (openSeq !== state.openSeq) return;
    refreshJobs();
    if (track.status === "error") { settled(st, `Render failed: ${track.error}`); return; }
    state.track = track;
    await showRendered(track);
    if (openSeq === state.openSeq && !resume) settled(st, `Rendered in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  }

  async function showRendered(track) {
    const st = $("render-status"), seq = ++state.renderSeq, openSeq = state.openSeq;
    measureRendered(track.id, seq);
    const g = track.rendered.grid;
    const isDown = g.target_beats.map((_, i) => state.tracks.a.isDown[i]);
    state.tracks.b = { duration: track.rendered.duration, beats: g.target_beats, isDown, barOf: barNumbers(isDown), peaks: track.rendered.peaks };
    $("rendered-card").hidden = false;
    const kbps = track.rendered.bitrate_kbps;
    $("rendered-title").textContent = `${g.target_bpm} BPM · ${fmtTime(track.rendered.duration)}${kbps ? ` · MP3 ${kbps} kbps` : ""} · grid = where each original beat now sits`;
    $("src-b").disabled = false;
    $("download").href = `/api/tracks/${track.id}/download`; $("download").hidden = false;
    working(st, "decoding the straightened audio…");
    try {
      await loadAudio("b", `/api/tracks/${track.id}/audio/rendered`);
      if (openSeq === state.openSeq) settled(st, "");
    } catch (e) { if (openSeq === state.openSeq) settled(st, "could not load the straightened audio"); }
    draw();
  }

  // The server re-detects the beats in the rendered audio after the MP3 is ready, so the
  // result turns up a little after the render itself.
  async function measureRendered(id, seq) {
    const st = $("analysis-b-status");
    working(st, "detecting beats in the output…");
    for (;;) {
      if (seq !== state.renderSeq) return;
      const a = state.track.rendered && state.track.rendered.analysis;
      if (a) {
        if (a.error) { settled(st, `could not measure: ${a.error}`); return; }
        settled(st, "");
        showRenderedStats(a);
        $("charts-b").hidden = false;
        draw();
        return;
      }
      await new Promise((r) => setTimeout(r, 1500));
      let t;
      try { t = await (await fetch(`/api/tracks/${id}`)).json(); } catch (e) { continue; }
      if (seq !== state.renderSeq) return;
      if (!t.rendered) { settled(st, ""); return; }
      state.track.rendered.analysis = t.rendered.analysis;
    }
  }

  function showRenderedStats(a) {
    const target = state.track.rendered.grid.target_beats;
    // how far each detected beat sits from the grid it was meant to land on
    const err = a.beats.map((b) => {
      const i = bsearch(target, b);
      return Math.min(Math.abs(b - (target[i] ?? Infinity)), Math.abs(b - (target[i - 1] ?? Infinity))) * 1000;
    }).sort((x, y) => x - y);
    const pct = (q) => err.length ? err[Math.min(err.length - 1, Math.floor(q * err.length))] : 0;
    const spread = a.max_bpm - a.min_bpm;
    $("stats-b").innerHTML = [
      ["Median tempo", a.median_bpm.toFixed(2), "BPM"],
      ["Tempo range", `${a.min_bpm.toFixed(1)}–${a.max_bpm.toFixed(1)}`, "BPM, 2nd–98th pct"],
      ["Drift", `±${(spread / 2 / a.median_bpm * 100).toFixed(1)}%`, `${spread.toFixed(1)} BPM spread`],
      ["Off the grid", `${pct(0.5).toFixed(0)} ms`, `median · 95% within ${pct(0.95).toFixed(0)} ms`],
      ["Beats", a.beats.length, `found again of ${target.length}`],
    ].map(statHtml).join("");
  }

  // ------------------------------------------------------------ job browser
  const fmtBytes = (n) => n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} kB`;
  function fmtAgo(seconds) {
    const s = Math.max(0, seconds);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min`;
    if (s < 86400 * 2) return `${Math.round(s / 3600)} h`;
    return `${Math.round(s / 86400)} d`;
  }
  const ENGINE = { r3: "R3", r2: "R2" };
  const BADGE = {
    analysing: ["detecting beats", "busy"],
    rendering: ["straightening", "busy"],
    measuring: ["measuring", "busy"],
    ready: ["beats found", ""],
    rendered: ["straightened", "done"],
    error: ["failed", "bad"],
  };
  let jobsTimer = 0;
  const LEVEL = { 2: " · half-time", 0.5: " · double-time" };

  // Refreshed on demand and after each upload or render, never on a timer: a poll
  // would keep the fly.io machine awake for as long as the tab is open.
  async function refreshJobs() {
    try {
      const res = await fetch("/api/tracks");
      if (!res.ok) throw new Error(res.status);
      showJobs(await res.json());
    } catch (e) { $("jobs-list").innerHTML = '<div class="hint">Could not list jobs.</div>'; }
  }

  function markOpenJob() {
    const id = state.track && state.track.id;
    document.querySelectorAll(".job").forEach((el) => el.classList.toggle("current", el.dataset.id === id));
  }

  function showJobs({ storage: s, jobs }) {
    const now = Date.now() / 1000;
    $("jobs-disk").textContent = `${fmtBytes(s.disk_free_bytes)} free of ${fmtBytes(s.disk_total_bytes)} · ` +
      `jobs are kept ${fmtAgo(s.max_age_seconds)} after their last change, or until the disk is under ${Math.round(s.min_free_fraction * 100)}% free`;
    const ORDER = ["straightened MP3", "straightened WAV", "upload", "playback WAV", "beat data"];
    const rank = (k) => { const i = ORDER.indexOf(k); return i < 0 ? ORDER.length : i; };
    $("jobs-list").innerHTML = jobs.length ? jobs.map((j) => {
      const links = [...j.files].sort((x, y) => rank(x.kind) - rank(y.kind)).map((f) =>
        `<span><a class="${f.kind === "straightened MP3" ? "main" : ""}" href="${esc(f.url)}" download="${esc(f.download_name)}" title="${esc(f.download_name)}">${esc(f.kind)}</a><small>${fmtBytes(f.bytes)}</small></span>`).join("");
      const r = j.render;
      const summary = [
        j.median_bpm ? `detected ${j.median_bpm.toFixed(1)} BPM` : "no beats yet",
        r && `straightened to ${r.target_bpm} BPM (${ENGINE[r.engine] || esc(r.engine)}${LEVEL[r.level] || ""})`,
      ].filter(Boolean).join(" · ");
      const [label, kind] = BADGE[j.status] || [j.status || "", ""];
      return `<div class="job" data-id="${esc(j.id)}" tabindex="0" role="button" title="Open this job">
        <div class="name">${esc(j.title || stem(j.name))} <span class="badge ${kind}">${kind === "busy" ? '<span class="spinner"></span>' : ""}${label}</span></div>
        <button class="del ghost" data-del="${esc(j.id)}" title="Delete this job and its files">✕</button>
        <div class="summary">${summary}</div>
        <div class="age">uploaded ${fmtAgo(now - j.created)} ago · deleted in ${fmtAgo(j.updated + s.max_age_seconds - now)}</div>
        <div class="links">${links}</div></div>`;
    }).join("") : '<div class="hint">No jobs yet. Drop a file above to start one.</div>';
    markOpenJob();
    // Keep the badges live only while a job is working. The work keeps the machine
    // awake anyway, and once it stops the page stops asking.
    clearTimeout(jobsTimer);
    if (jobs.some((j) => (BADGE[j.status] || [])[1] === "busy")) jobsTimer = setTimeout(refreshJobs, 3000);
  }

  $("jobs-list").addEventListener("click", async (e) => {
    if (e.target.closest("a")) return; // a download link, not a request to open
    const del = e.target.closest("[data-del]");
    if (del) {
      const id = del.dataset.del;
      if (!confirm("Delete this job and all its files?")) return;
      const res = await fetch(`/api/tracks/${id}`, { method: "DELETE" });
      if (!res.ok) { let msg = await res.text(); try { msg = JSON.parse(msg).detail || msg; } catch (err) {} alert(msg); return; }
      if (state.track && state.track.id === id) { resetView(); status(""); history.replaceState(null, "", location.pathname); }
      refreshJobs();
      return;
    }
    const job = e.target.closest(".job");
    if (job && !(state.track && state.track.id === job.dataset.id)) openJob(job.dataset.id);
  });
  $("jobs-list").addEventListener("keydown", (e) => {
    const job = e.target.closest(".job");
    if (job && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openJob(job.dataset.id); }
  });

  // -------------------------------------------------------------- wiring
  $("file").addEventListener("change", (e) => { if (e.target.files[0]) upload(e.target.files[0]); });
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) upload(f); });

  $("render").addEventListener("click", () => renderTrack());
  $("redetect").addEventListener("click", redetect);
  $("bpm").addEventListener("input", draw);
  $("level").addEventListener("change", () => {
    const a = state.track && state.track.analysis; if (!a) return;
    $("bpm").value = Math.round(a.median_bpm * parseFloat($("level").value));
    draw();
  });
  $("bpm-median").addEventListener("click", () => { $("bpm").value = (state.track.analysis.median_bpm * parseFloat($("level").value)).toFixed(2); draw(); });
  $("bpm-round").addEventListener("click", () => { $("bpm").value = Math.round(parseFloat($("bpm").value)); draw(); });
  $("play").addEventListener("click", () => { state.playing ? pause() : play(); });
  $("stop").addEventListener("click", () => { pause(); state.pausedAt = 0; draw(); });
  $("src-a").addEventListener("click", () => switchSource("a"));
  $("src-b").addEventListener("click", () => switchSource("b"));
  $("click").addEventListener("change", () => { if (state.playing) startClicks(); });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); state.playing ? pause() : play(); }
    if (e.key === "a" || e.key === "A") switchSource("a");
    if (e.key === "b" || e.key === "B") switchSource("b");
  });
  window.addEventListener("resize", draw);
  $("jobs-refresh").addEventListener("click", refreshJobs);
  wireWave("a"); wireWave("b");
  health();
  refreshJobs();
  const linked = /^[0-9a-f]{12}$/.exec(new URLSearchParams(location.search).get("job") || "");
  if (linked) openJob(linked[0]);
})();
