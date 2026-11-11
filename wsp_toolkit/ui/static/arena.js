/* WSP Arena: a deterministic canvas renderer for challenge runs.
 *
 * render(t) is a pure function of (timeline, t): the same frame comes out every time, so live viewing, replays and
 * frame-exact video export (wsp arena record) all share this code. The timeline comes from wsp_toolkit/core/arena.py.
 *
 * The viewer sees the hidden ground truth (damaged heads, planted triggers); the agent never does.
 */
(function (global) {
  'use strict';

  const C = {
    bg: '#03050a', text: '#e6edf7', muted: '#8a97ad', dim: '#56627a',
    cyan: '#22d3ee', violet: '#a78bfa', magenta: '#e879f9', lime: '#a3e635', amber: '#fbbf24', red: '#f43f5e',
    sky: '#38bdf8', white: '#ffffff',
  };
  const DISPLAY = '"Orbitron", "Inter", system-ui, sans-serif';
  const MONO = '"JetBrains Mono", "DejaVu Sans Mono", ui-monospace, monospace';
  const SANS = '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif';

  // ------------------------------------------------------------------ math helpers
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, k) => a + (b - a) * k;
  const easeOut = k => 1 - Math.pow(1 - clamp(k), 3);
  const easeInOut = k => { k = clamp(k); return k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2; };
  const easeBack = k => { k = clamp(k); const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(k - 1, 3) + c1 * Math.pow(k - 1, 2); };
  const seg = (p, a, b) => clamp((p - a) / (b - a));
  function hash(n) { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453123; return x - Math.floor(x); }
  function hex2rgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  const _rgbCache = {};
  function rgba(h, a) { const c = _rgbCache[h] || (_rgbCache[h] = hex2rgb(h)); return `rgba(${c[0]},${c[1]},${c[2]},${clamp(a)})`; }
  function mix(h1, h2, k) {
    const a = hex2rgb(h1), b = hex2rgb(h2); k = clamp(k);
    const c = a.map((v, i) => Math.round(lerp(v, b[i], k)));
    return '#' + c.map(v => v.toString(16).padStart(2, '0')).join('');
  }
  function scoreColor(s) { return s == null ? C.muted : s >= 0.9 ? C.lime : s >= 0.6 ? C.cyan : s >= 0.3 ? C.amber : C.red; }
  function fmt(x, d = 3) { return x == null ? '—' : Number(x).toFixed(d); }
  function clock(sec) {
    sec = Math.max(0, sec); const m = Math.floor(sec / 60), s = sec - m * 60;
    return String(m).padStart(2, '0') + ':' + s.toFixed(1).padStart(4, '0');
  }

  // ------------------------------------------------------------------ timeline pre-processing
  function headKey(l, h) { return l + '.' + h; }

  function precompute(tl) {
    const beats = tl.beats || [];
    const states = [];
    let st = { applied: [], patches: {}, budget: 0, tools: 0, scores: [], log: [], caption: null, tiles: {},
               edits: 0, lastTokens: null, errors: 0 };
    for (let i = 0; i < beats.length; i++) {
      const b = beats[i];
      st = { ...st, applied: st.applied.slice(), patches: { ...st.patches }, scores: st.scores.slice(),
             log: st.log.slice(), tiles: { ...st.tiles } };
      if (b.budget_used != null) st.budget = b.budget_used;
      if (b.tools != null) st.tools = b.tools;
      if (b.tokens && b.tokens.length) st.lastTokens = b.tokens;
      const line = logLine(b);
      if (line) st.log.push({ ...line, t0: b.t, rt: b.rt, ok: b.ok !== false });
      if (b.type === 'edit') {
        st.patches[b.patch_id] = b.targets || [];
        st.applied = st.applied.filter(p => p !== b.patch_id).concat([b.patch_id]);
        if (!b.reapply) st.edits += 1;
        if (b.text) st.caption = { text: b.text, label: 'EDIT RATIONALE', t0: b.t + 0.9, color: C.lime };
      } else if (b.type === 'revert') {
        st.applied = st.applied.filter(p => p !== b.patch_id);
      } else if (b.type === 'practice' && b.score != null) {
        st.scores.push({ score: b.score, split: b.split });
      } else if (b.type === 'note') {
        const label = b.src === 'thinking' ? 'AGENT · THINKING' : 'AGENT · SAYS';
        st.caption = { text: b.text, label, t0: b.t, color: b.src === 'thinking' ? C.violet : C.cyan };
      } else if (b.type === 'error') {
        st.errors += 1;
      }
      if (b.tile != null && b.type !== 'note') {
        const prev = st.tiles[b.tile] || {};
        st.tiles[b.tile] = { ...prev, touched: true, errors: (prev.errors || 0) + (b.type === 'error' ? 1 : 0),
                             edited: prev.edited || b.type === 'edit' };
      }
      states.push(st);
    }
    return states;
  }

  function targetText(targets) {
    if (!targets || !targets.length) return '';
    const heads = [];
    const other = new Set();
    for (const t of targets) {
      if (t.kind === 'heads') {
        if (t.heads) t.heads.forEach(h => heads.push(headKey(t.layer, h)));
        else other.add('L' + t.layer + ' ' + (t.proj || '') + '_proj');
      } else if (t.kind === 'mlp') other.add('layer ' + t.layer + (t.inputs ? ' inputs ' + t.inputs.join(',') : ''));
      else other.add(t.kind + (t.row != null ? ' row t' + t.row : ''));
    }
    const parts = [];
    if (heads.length) parts.push('heads ' + [...new Set(heads)].join(' '));
    other.forEach(o => parts.push(o));
    return parts.join(' · ');
  }

  function logLine(b) {
    switch (b.type) {
      case 'scan': return { tool: b.tool, detail: b.label || '', color: C.cyan };
      case 'forward': return { tool: b.tool, detail: (b.tokens && b.tokens.length ? b.tokens.length + ' tokens' : ''), color: C.sky };
      case 'attention': return { tool: b.tool, detail: 'layers [' + (b.layers || []).join(',') + ']', color: C.violet };
      case 'practice': return { tool: b.tool, detail: (b.split || '') + (b.score != null ? ' score ' + fmt(b.score) : ''), color: scoreColor(b.score) };
      case 'edit': return { tool: b.tool, detail: ((b.ops || []).join('+') + ' ' + targetText(b.targets)).trim() + (b.params ? ' · ' + b.params.toLocaleString('en-US') + ' params' : ''), color: C.lime };
      case 'revert': return { tool: b.tool, detail: b.patch_id || '', color: C.amber };
      case 'error': return { tool: b.tool, detail: b.text || 'error', color: C.red };
      case 'diff': return { tool: b.tool, detail: b.label || '', color: C.magenta };
      case 'submit': return { tool: 'challenge_submit', detail: 'answer locked in', color: C.white };
      case 'generic': return { tool: b.tool, detail: '', color: C.muted };
      default: return null;
    }
  }

  function headStatus(key, st, initial) {
    const dmg = (initial.damaged || {})[key];
    const [l, h] = key.split('.').map(Number);
    const hits = [];
    for (const pid of st.applied) for (const t of (st.patches[pid] || [])) {
      if (t.kind === 'heads' && t.layer === l && (!t.heads || t.heads.includes(h))) hits.push(t);
    }
    if (!hits.length) return dmg ? (dmg === 'flip' ? 'flip' : 'damaged') : 'ok';
    if (dmg) {
      const fixes = hits.some(t => t.op !== 'scale' || t.factor == null || (dmg === 'flip' ? t.factor < 0 : t.factor > 1));
      return fixes ? 'patched' : 'changed';
    }
    return 'changed';
  }

  function partTouched(kind, st) {
    for (const pid of st.applied) for (const t of (st.patches[pid] || [])) if (t.kind === kind) return t;
    return null;
  }

  const STATUS = {
    ok: { color: C.cyan, label: '' },
    damaged: { color: C.red, label: 'DAMAGED' },
    flip: { color: C.magenta, label: 'SIGN-FLIPPED' },
    patched: { color: C.lime, label: 'PATCHED' },
    changed: { color: C.amber, label: 'CHANGED' },
    induction: { color: C.violet, label: 'INDUCTION' },
    previous: { color: C.sky, label: 'PREV-TOKEN' },
    none: { color: C.dim, label: '' },
  };

  // ------------------------------------------------------------------ layout
  function layout(W, H) {
    const land = W / H >= 1.45;
    const u = land ? H / 1080 : W / 1080;
    const pad = 34 * u;
    const header = { x: pad, y: pad * 0.7, w: W - 2 * pad, h: 64 * u };
    const top = header.y + header.h + 16 * u;
    if (land) {
      const side = 410 * u, gap = 26 * u;
      const agent = { x: pad, y: top, w: side, h: 238 * u };
      const caption = { x: pad, y: agent.y + agent.h + gap * 0.7, w: side, h: H - pad - (agent.y + agent.h + gap * 0.7) };
      const telem = { x: W - pad - side, y: top, w: side, h: H - pad - top };
      const cx0 = pad + side + gap, cx1 = W - pad - side - gap;
      const log = { x: cx0, y: H - pad - 196 * u, w: cx1 - cx0, h: 196 * u };
      const stage = { x: cx0, y: top, w: cx1 - cx0, h: log.y - top - gap * 0.7 };
      return { land, u, pad, W, H, header, agent, caption, telem, log, stage, compact: false };
    }
    const stageH = H * 0.47;
    const stage = { x: pad, y: top, w: W - 2 * pad, h: stageH };
    const gap = 16 * u;
    const logH = Math.max(128 * u, H * 0.1);
    const log = { x: pad, y: H - pad - logH, w: W - 2 * pad, h: logH };
    const midY = stage.y + stage.h + gap, midH = log.y - gap - midY;
    const colW = (W - 2 * pad - gap) / 2;
    const agent = { x: pad, y: midY, w: colW, h: 118 * u };
    const caption = { x: pad, y: agent.y + agent.h + gap * 0.6, w: colW, h: midH - agent.h - gap * 0.6 };
    const telem = { x: pad + colW + gap, y: midY, w: colW, h: midH };
    return { land, u, pad, W, H, header, agent, caption, telem, log, stage, compact: true };
  }

  // ------------------------------------------------------------------ drawing helpers
  function chamfer(ctx, r, c) {
    ctx.beginPath();
    ctx.moveTo(r.x + c, r.y); ctx.lineTo(r.x + r.w, r.y); ctx.lineTo(r.x + r.w, r.y + r.h - c);
    ctx.lineTo(r.x + r.w - c, r.y + r.h); ctx.lineTo(r.x, r.y + r.h); ctx.lineTo(r.x, r.y + c); ctx.closePath();
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function text(ctx, s, x, y, o = {}) {
    ctx.font = `${o.weight || 400} ${o.size || 16}px ${o.font || SANS}`;
    ctx.fillStyle = o.color || C.text;
    ctx.textAlign = o.align || 'left';
    ctx.textBaseline = o.base || 'alphabetic';
    if ('letterSpacing' in ctx) ctx.letterSpacing = (o.spacing || 0) + 'px';
    if (o.glow) { ctx.shadowColor = o.glowColor || o.color || C.cyan; ctx.shadowBlur = o.glow; }
    ctx.globalAlpha = o.alpha == null ? 1 : clamp(o.alpha);
    ctx.fillText(s, x, y);
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
  }
  function measure(ctx, s, o = {}) {
    ctx.font = `${o.weight || 400} ${o.size || 16}px ${o.font || SANS}`;
    if ('letterSpacing' in ctx) ctx.letterSpacing = (o.spacing || 0) + 'px';
    const w = ctx.measureText(s).width;
    if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
    return w;
  }
  const _wrapCache = new Map();
  function wrap(ctx, s, maxW, o) {
    const key = s + '|' + maxW + '|' + (o.size || 16) + (o.font || '');
    if (_wrapCache.has(key)) return _wrapCache.get(key);
    const words = String(s).split(/\s+/);
    const lines = []; let line = '';
    for (const w of words) {
      const test = line ? line + ' ' + w : w;
      if (measure(ctx, test, o) > maxW && line) { lines.push(line); line = w; } else line = test;
    }
    if (line) lines.push(line);
    if (_wrapCache.size > 400) _wrapCache.clear();
    _wrapCache.set(key, lines);
    return lines;
  }
  function panel(ctx, r, title, L, alpha = 1, accent = C.cyan) {
    const u = L.u, c = 16 * u;
    ctx.save();
    ctx.globalAlpha = alpha;
    chamfer(ctx, r, c);
    const g = ctx.createLinearGradient(r.x, r.y, r.x, r.y + r.h);
    g.addColorStop(0, 'rgba(16,24,44,0.80)'); g.addColorStop(1, 'rgba(8,12,24,0.72)');
    ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = 'rgba(120,200,255,0.16)'; ctx.lineWidth = 1.2 * u; ctx.stroke();
    // accent corner
    ctx.strokeStyle = accent; ctx.lineWidth = 2 * u; ctx.shadowColor = accent; ctx.shadowBlur = 8 * u;
    ctx.beginPath(); ctx.moveTo(r.x, r.y + c + 26 * u); ctx.lineTo(r.x, r.y + c); ctx.lineTo(r.x + c, r.y); ctx.lineTo(r.x + c + 26 * u, r.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(r.x + r.w, r.y + r.h - c - 26 * u); ctx.lineTo(r.x + r.w, r.y + r.h - c); ctx.lineTo(r.x + r.w - c, r.y + r.h); ctx.lineTo(r.x + r.w - c - 26 * u, r.y + r.h); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.restore();
    if (title) {
      ctx.save(); ctx.globalAlpha = alpha;
      ctx.fillStyle = accent; ctx.fillRect(r.x + 20 * u, r.y + 22 * u, 7 * u, 7 * u);
      ctx.restore();
      text(ctx, title, r.x + 36 * u, r.y + 30 * u, { font: MONO, size: 13 * u, color: accent, spacing: 2.5 * u, weight: 700, alpha });
    }
  }
  function pill(ctx, s, x, y, color, L, o = {}) {
    const u = L.u, size = (o.size || 12) * u;
    const w = measure(ctx, s, { font: MONO, size, spacing: 1.2 * u, weight: 700 }) + 18 * u, h = size + 12 * u;
    roundRect(ctx, x, y, w, h, h / 2);
    ctx.fillStyle = rgba(color, 0.12 * (o.alpha ?? 1)); ctx.fill();
    ctx.strokeStyle = rgba(color, 0.55 * (o.alpha ?? 1)); ctx.lineWidth = 1.1 * u; ctx.stroke();
    text(ctx, s, x + 9 * u, y + h / 2 + 0.5 * u, { font: MONO, size, color, spacing: 1.2 * u, weight: 700, base: 'middle', alpha: o.alpha });
    return w;
  }

  // ------------------------------------------------------------------ background + overlays
  let _noise = null;
  function noiseCanvas() {
    if (_noise) return _noise;
    const n = document.createElement('canvas'); n.width = n.height = 256;
    const g = n.getContext('2d'); const img = g.createImageData(256, 256);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = Math.floor(hash(i * 0.37) * 255); img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0); _noise = n; return n;
  }

  function background(ctx, L, t, intensity) {
    const { W, H, u } = L;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    // drifting nebula glows
    const g1x = W * (0.3 + 0.05 * Math.sin(t * 0.11)), g1y = H * 0.15;
    let g = ctx.createRadialGradient(g1x, g1y, 0, g1x, g1y, W * 0.55);
    g.addColorStop(0, `rgba(34,211,238,${0.10 * intensity})`); g.addColorStop(1, 'rgba(34,211,238,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    const g2x = W * (0.75 + 0.04 * Math.cos(t * 0.09)), g2y = H * 0.85;
    g = ctx.createRadialGradient(g2x, g2y, 0, g2x, g2y, W * 0.5);
    g.addColorStop(0, `rgba(167,139,250,${0.10 * intensity})`); g.addColorStop(1, 'rgba(167,139,250,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // perspective floor grid
    const horizon = H * 0.58, vpx = W / 2;
    ctx.save();
    ctx.globalAlpha = 0.55 * intensity;
    ctx.beginPath(); ctx.rect(0, horizon, W, H - horizon); ctx.clip();
    const fg = ctx.createLinearGradient(0, horizon, 0, H);
    fg.addColorStop(0, 'rgba(34,211,238,0)'); fg.addColorStop(1, 'rgba(34,211,238,0.22)');
    ctx.strokeStyle = fg; ctx.lineWidth = 1 * u;
    for (let i = -24; i <= 24; i++) {
      ctx.beginPath(); ctx.moveTo(vpx + i * 12 * u, horizon); ctx.lineTo(vpx + i * 150 * u, H); ctx.stroke();
    }
    const scroll = (t * 0.35) % 1;
    for (let k = 0; k < 18; k++) {
      const z = (k + scroll) / 18; const y = horizon + Math.pow(z, 2.2) * (H - horizon);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.restore();
    // dust
    for (let i = 0; i < 70; i++) {
      const x = ((hash(i) * W) + t * (6 + 10 * hash(i + 9)) * u) % W;
      const y = (hash(i + 3) * H + Math.sin(t * 0.3 + i) * 6 * u);
      const a = (0.15 + 0.35 * hash(i + 5)) * (0.6 + 0.4 * Math.sin(t * (1 + hash(i + 7)) + i)) * intensity;
      ctx.fillStyle = rgba(i % 5 === 0 ? C.violet : C.cyan, a);
      ctx.fillRect(x, y, 1.6 * u, 1.6 * u);
    }
  }

  function overlays(ctx, L, t, frame) {
    const { W, H, u } = L;
    // scanlines
    ctx.fillStyle = 'rgba(0,0,0,0.13)';
    for (let y = 0; y < H; y += Math.max(3, Math.round(3 * u))) ctx.fillRect(0, y, W, Math.max(1, u));
    // a slow rolling bright band
    const by = ((t * 0.08) % 1.2 - 0.1) * H;
    const bg = ctx.createLinearGradient(0, by - 60 * u, 0, by + 60 * u);
    bg.addColorStop(0, 'rgba(120,220,255,0)'); bg.addColorStop(0.5, 'rgba(120,220,255,0.035)'); bg.addColorStop(1, 'rgba(120,220,255,0)');
    ctx.fillStyle = bg; ctx.fillRect(0, by - 60 * u, W, 120 * u);
    // grain
    ctx.save(); ctx.globalAlpha = 0.045; ctx.globalCompositeOperation = 'overlay';
    const n = noiseCanvas(); const ox = Math.floor(hash(frame) * 256), oy = Math.floor(hash(frame + 1) * 256);
    ctx.translate(-ox, -oy); ctx.fillStyle = ctx.createPattern(n, 'repeat'); ctx.fillRect(0, 0, W + 256, H + 256);
    ctx.restore();
    // vignette
    const v = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
    v.addColorStop(0, 'rgba(0,0,0,0)'); v.addColorStop(1, 'rgba(0,0,0,0.55)');
    ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
  }

  // ------------------------------------------------------------------ the arena
  function create(canvas, timeline, opts = {}) {
    const ctx = canvas.getContext('2d');
    let tl = timeline, states = precompute(tl);
    let playing = false, t = 0, last = null, raf = null, speed = 1, live = !!opts.live, onTick = null;

    function beatAt(time) {
      const beats = tl.beats || [];
      let lo = 0, hi = beats.length - 1, idx = -1;
      while (lo <= hi) { const mid = (lo + hi) >> 1; if (beats[mid].t <= time) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
      return idx;
    }

    function render(time) {
      const W = canvas.width, H = canvas.height;
      const L = layout(W, H), u = L.u;
      const frame = Math.floor(time * 30);
      const beats = tl.beats || [];
      const idx = beatAt(time);
      const b = idx >= 0 ? beats[idx] : null;
      const inBeat = b && time < b.t + b.dur;
      const lt = b ? time - b.t : 0, p = b ? clamp(lt / b.dur) : 0;
      const prev = idx > 0 ? states[idx - 1] : emptyState();
      const cur = idx >= 0 ? states[idx] : emptyState();
      const intro = b && b.type === 'intro' && inBeat;
      const hud = intro ? easeOut(seg(lt, 3.3, 4.4)) : 1;
      const S = { time, L, u, b: inBeat ? b : null, lt, p, prev, cur, hud, frame, idx };

      ctx.save();
      // camera shake on impact moments
      const shake = impact(S);
      if (shake) ctx.translate((hash(frame) - 0.5) * shake * u, (hash(frame + 3) - 0.5) * shake * u);
      background(ctx, L, time, intro ? easeOut(seg(lt, 0.4, 2.0)) : 1);
      if (hud > 0) {
        drawStage(ctx, S);
        drawHeader(ctx, S);
        drawAgent(ctx, S);
        drawCaption(ctx, S);
        drawTelemetry(ctx, S);
        drawLog(ctx, S);
      }
      if (inBeat) {
        if (b.type === 'intro') drawIntro(ctx, S);
        else if (b.type === 'submit') drawSubmit(ctx, S);
        else if (b.type === 'outro') { drawSubmit(ctx, { ...S, p: 1, lt: 99, outro: true }); drawOutro(ctx, S); }
        else if (b.type === 'error') drawErrorFlash(ctx, S);
        else if (b.type === 'revert') drawRewind(ctx, S);
      } else if (!tl.run.finished && time >= tl.duration) {
        drawAwaiting(ctx, S);
      }
      ctx.restore();
      overlays(ctx, L, time, frame);
      if (inBeat && b.type === 'outro') {
        const f = seg(lt, b.dur - 0.7, b.dur);
        if (f > 0) { ctx.fillStyle = `rgba(0,0,0,${f})`; ctx.fillRect(0, 0, W, H); }
      }
    }

    function emptyState() {
      return { applied: [], patches: {}, budget: 0, tools: 0, scores: [], log: [], caption: null, tiles: {}, edits: 0, lastTokens: null, errors: 0 };
    }

    function impact(S) {
      const b = S.b; if (!b) return 0;
      if (b.type === 'edit' && !b.reapply) { const k = seg(S.p, 0.55, 0.75); return k > 0 && k < 1 ? 10 * (1 - k) : 0; }
      if (b.type === 'error') return S.lt < 0.35 ? 8 * (1 - S.lt / 0.35) : 0;
      if (b.type === 'submit') { const k = seg(S.lt, 3.0, 3.5); return k > 0 && k < 1 ? 7 * (1 - k) : 0; }
      return 0;
    }

    // ---------------------------------------------------------------- header
    function drawHeader(ctx, S) {
      const { L, u, time, hud } = S, r = L.header, run = tl.run;
      ctx.save(); ctx.globalAlpha = hud;
      // logo
      const lx = r.x, ly = r.y + r.h / 2;
      drawLogo(ctx, lx + 22 * u, ly, 20 * u, time);
      text(ctx, 'WSP', lx + 54 * u, ly + 9 * u, { font: DISPLAY, size: 26 * u, weight: 800, color: C.text, spacing: 3 * u, alpha: hud });
      const w1 = measure(ctx, 'WSP', { font: DISPLAY, size: 26 * u, weight: 800, spacing: 3 * u });
      text(ctx, 'ARENA', lx + 60 * u + w1, ly + 9 * u, { font: DISPLAY, size: 26 * u, weight: 600, color: C.cyan, spacing: 3 * u, glow: 12 * u, alpha: hud });
      // challenge title (centre)
      const cx = L.land ? L.stage.x + L.stage.w / 2 : r.x + r.w * 0.52;
      const title = run.title.toUpperCase();
      const tsize = (L.land ? 24 : 17) * u;
      text(ctx, title, cx, ly + 2 * u, { font: DISPLAY, size: tsize, weight: 600, align: 'center', color: C.text, spacing: 2 * u, alpha: hud });
      text(ctx, `L${run.level} · ${run.kind.toUpperCase()} · SEED ${run.seed}`, cx, ly + 24 * u,
           { font: MONO, size: 12 * u, align: 'center', color: C.muted, spacing: 2 * u, alpha: hud });
      // live / replay indicator + clock
      const rx = r.x + r.w;
      const isLive = live && !run.finished;
      const blink = 0.5 + 0.5 * Math.sin(time * 5);
      ctx.fillStyle = rgba(isLive ? C.red : C.amber, 0.4 + 0.6 * blink);
      const tcW = measure(ctx, 'T+' + clock(time), { font: MONO, size: 14 * u, weight: 700, spacing: 1 * u });
      const lab = isLive ? 'LIVE' : 'REPLAY';
      const labW = measure(ctx, lab, { font: MONO, size: 14 * u, weight: 700, spacing: 3 * u });
      const lx0 = rx - tcW - 22 * u - labW;
      ctx.beginPath(); ctx.arc(lx0 - 12 * u, ly - 6 * u, 6 * u, 0, Math.PI * 2); ctx.fill();
      text(ctx, lab, lx0, ly - 1 * u, { font: MONO, size: 14 * u, weight: 700, color: isLive ? C.red : C.amber, spacing: 3 * u, alpha: hud });
      text(ctx, 'T+' + clock(time), rx, ly - 1 * u, { font: MONO, size: 14 * u, weight: 700, color: C.text, align: 'right', spacing: 1 * u, alpha: hud });
      const rt = S.b && S.b.rt != null ? S.b.rt : (S.cur.log.length ? S.cur.log[S.cur.log.length - 1].rt : 0);
      text(ctx, 'RUN CLOCK ' + clock(rt || 0), rx, ly + 20 * u, { font: MONO, size: 11 * u, color: C.muted, align: 'right', spacing: 1.5 * u, alpha: hud });
      // hairline
      const hg = ctx.createLinearGradient(r.x, 0, r.x + r.w, 0);
      hg.addColorStop(0, 'rgba(34,211,238,0)'); hg.addColorStop(0.5, 'rgba(34,211,238,0.45)'); hg.addColorStop(1, 'rgba(34,211,238,0)');
      ctx.fillStyle = hg; ctx.fillRect(r.x, r.y + r.h + 4 * u, r.w, 1.2 * u);
      ctx.restore();
    }

    function drawLogo(ctx, x, y, s, time) {
      ctx.save(); ctx.translate(x, y); ctx.rotate(time * 0.4);
      ctx.strokeStyle = C.cyan; ctx.lineWidth = 2 * s / 20; ctx.shadowColor = C.cyan; ctx.shadowBlur = 10 * s / 20;
      ctx.beginPath();
      for (let i = 0; i < 6; i++) { const a = Math.PI / 3 * i; const px = Math.cos(a) * s, py = Math.sin(a) * s; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.closePath(); ctx.stroke();
      ctx.rotate(-time * 0.8);
      ctx.fillStyle = C.violet; ctx.shadowColor = C.violet;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        const k = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(time * 3 + i * 1.7 + j * 2.3));
        ctx.globalAlpha = k; ctx.fillRect((i - 1) * s * 0.36 - s * 0.12, (j - 1) * s * 0.36 - s * 0.12, s * 0.24, s * 0.24);
      }
      ctx.restore();
    }

    // ---------------------------------------------------------------- agent card
    function drawAgent(ctx, S) {
      const { L, u, hud, time } = S, r = L.agent, run = tl.run;
      panel(ctx, r, 'AGENT UNDER TEST', L, hud, C.cyan);
      const x = r.x + 22 * u; let y = r.y + 76 * u;
      const scripted = /scripted/i.test(run.model || '');
      const name = String(run.model || 'unknown').toUpperCase();
      let size = (L.compact ? 24 : 30) * u;
      if (L.compact) y -= 12 * u;
      while (measure(ctx, name, { font: DISPLAY, size, weight: 800, spacing: 1.5 * u }) > r.w - 44 * u && size > 12 * u) size -= 1 * u;
      text(ctx, name, x, y, { font: DISPLAY, size, weight: 800, color: C.text, spacing: 1.5 * u, glow: 14 * u, glowColor: C.cyan, alpha: hud });
      y += (L.compact ? 22 : 26) * u;
      let harness = scripted ? 'deterministic script · no AI' : (run.harness || '');
      while (measure(ctx, harness, { font: MONO, size: 13 * u }) > r.w - 44 * u && harness.length > 8) harness = harness.slice(0, -2);
      text(ctx, harness, x, y, { font: MONO, size: 13 * u, color: scripted ? C.amber : C.muted, spacing: 0.5 * u, alpha: hud });
      if (L.compact) {
        y += 14 * u;
        let qx = x;
        qx += pill(ctx, run.access.toUpperCase(), qx, y, run.access === 'weights_only' ? C.magenta : C.cyan, L, { size: 10, alpha: hud }) + 6 * u;
        if (qx + 100 * u < r.x + r.w) pill(ctx, run.budget_per_session ? run.budget_per_session + ' PASSES' : 'ZERO PASSES', qx, y, run.budget_per_session ? C.sky : C.red, L, { size: 10, alpha: hud });
        return;
      }
      // activity bars
      const busy = S.b && !['intro', 'submit', 'outro'].includes(S.b.type) ? 1 : 0.25;
      const bars = Math.floor((r.w - 44 * u) / (13 * u));
      y += 46 * u;
      for (let i = 0; i < bars; i++) {
        const hgt = (4 + (L.compact ? 12 : 22) * Math.abs(Math.sin(time * (2.2 + hash(i) * 3) + i)) * busy) * u;
        ctx.fillStyle = rgba(i % 7 === 3 ? C.violet : C.cyan, (0.35 + 0.5 * busy) * hud);
        ctx.fillRect(x + i * 13 * u, y - hgt, 7 * u, hgt);
      }
      y += (L.compact ? 14 : 22) * u;
      let px = x;
      px += pill(ctx, run.access.toUpperCase(), px, y, run.access === 'weights_only' ? C.magenta : C.cyan, L, { size: 11, alpha: hud }) + 8 * u;
      if (px + 120 * u < r.x + r.w) pill(ctx, run.budget_per_session ? run.budget_per_session + ' PASSES' + (run.sessions > 1 ? ' × ' + run.sessions : '') : 'ZERO PASSES', px, y,
           run.budget_per_session ? C.sky : C.red, L, { size: 11, alpha: hud });
    }

    // ---------------------------------------------------------------- caption (rationale / thoughts)
    function drawCaption(ctx, S) {
      const { L, u, hud, time, cur } = S, r = L.caption, run = tl.run;
      const cap = cur.caption;
      panel(ctx, r, cap ? cap.label : 'MISSION BRIEF', L, hud, cap ? cap.color : C.violet);
      const x = r.x + 22 * u, y0 = r.y + 66 * u;
      if (!cap) {
        const size = (L.compact ? 15 : 18) * u, lh = size * 1.45;
        const lines = wrap(ctx, run.summary || '', r.w - 44 * u, { size });
        lines.forEach((ln, i) => text(ctx, ln, x, y0 + i * lh, { size, color: C.text, alpha: hud * 0.9 }));
        let y = y0 + lines.length * lh + 14 * u;
        if (run.scoring && !L.compact) {
          text(ctx, 'SCORING', x, y, { font: MONO, size: 11 * u, color: C.muted, spacing: 2 * u, alpha: hud });
          y += 22 * u;
          wrap(ctx, run.scoring, r.w - 44 * u, { size: 14 * u }).slice(0, 6).forEach((ln, i) =>
            text(ctx, ln, x, y + i * 20 * u, { size: 14 * u, color: C.muted, alpha: hud }));
        }
        return;
      }
      const size = (L.compact ? 17 : 21) * u, lh = size * 1.42;
      const shown = Math.floor(Math.max(0, time - cap.t0) * 80);
      const full = '“' + cap.text + '”';
      const lines = wrap(ctx, full, r.w - 44 * u, { size, weight: 500 });
      const maxLines = Math.max(1, Math.floor((r.h - 84 * u) / lh));
      let left = shown;
      lines.slice(0, maxLines).forEach((ln, i) => {
        if (left <= 0) return;
        let s = ln.slice(0, left); left -= ln.length + 1;
        if (i === maxLines - 1 && lines.length > maxLines && left > 0) s = s.replace(/\s*\S*$/, ' …');
        text(ctx, s, x, y0 + i * lh, { size, weight: 500, color: C.text, alpha: hud });
        if (left <= 0 && Math.floor(time * 3) % 2 === 0) {
          const w = measure(ctx, s, { size, weight: 500 });
          ctx.fillStyle = rgba(cap.color, hud); ctx.fillRect(x + w + 3 * u, y0 + i * lh - size * 0.8, 3 * u, size);
        }
      });
    }

    // ---------------------------------------------------------------- telemetry
    function drawTelemetry(ctx, S) {
      const { L, u, hud, cur, prev, b, p, time } = S, r = L.telem;
      panel(ctx, r, 'TELEMETRY', L, hud, C.violet);
      const x = r.x + 22 * u, w = r.w - 44 * u;
      let y = r.y + 62 * u;
      // budget
      const total = tl.budget_total || 0;
      let used = cur.budget;
      if (b && b.budget_used != null && (b.type === 'practice' || b.type === 'forward' || b.type === 'attention')) {
        used = lerp(prev.budget, b.budget_used, easeOut(seg(p, 0.1, 0.6)));
      }
      text(ctx, 'FORWARD-PASS BUDGET', x, y, { font: MONO, size: 12 * u, color: C.muted, spacing: 2 * u, alpha: hud });
      const remain = Math.max(0, total - used);
      text(ctx, total ? `${Math.round(remain)}` : '0', x + w, y + 2 * u, { font: DISPLAY, size: 22 * u, weight: 700, align: 'right', color: total ? (remain / total < 0.2 ? C.red : C.text) : C.red, alpha: hud });
      y += 14 * u;
      const segs = Math.min(total || 24, 50);
      const sw = w / segs;
      for (let i = 0; i < segs; i++) {
        const frac = total ? (i + 1) / segs : 1;
        const isUsed = !total || frac * total > remain + 1e-6;
        const burning = !isUsed && false;
        ctx.fillStyle = isUsed ? 'rgba(80,90,110,0.35)' : rgba(remain / Math.max(1, total) < 0.2 ? C.red : C.sky, 0.85 * hud);
        if (!isUsed || burning) { ctx.shadowColor = C.sky; ctx.shadowBlur = 6 * u; }
        ctx.fillRect(x + i * sw + 1 * u, y, sw - 2 * u, 16 * u);
        ctx.shadowBlur = 0;
      }
      if (!total) text(ctx, 'WEIGHTS ONLY · FLYING BLIND', x + w / 2, y + 12 * u, { font: MONO, size: 11 * u, weight: 700, align: 'center', color: C.red, spacing: 2 * u, alpha: hud * (0.6 + 0.4 * Math.sin(time * 4)) });
      // sparks when budget burns
      if (b && b.budget_used != null && b.budget_used > prev.budget && p < 0.7 && total) {
        const ex = x + (remain / total) * w;
        for (let i = 0; i < 14; i++) {
          const a = hash(i + idxSeed(b)) * Math.PI - Math.PI, sp = (20 + 50 * hash(i + 2)) * u, k = (p * 6 + hash(i)) % 1;
          ctx.fillStyle = rgba(i % 2 ? C.amber : C.sky, (1 - k) * hud);
          ctx.fillRect(ex + Math.cos(a) * sp * k, y + 8 * u + Math.sin(a) * sp * k, 2.5 * u, 2.5 * u);
        }
      }
      text(ctx, total ? `${Math.round(used)} of ${total} used` : 'no forward passes allowed', x, y + 38 * u, { font: MONO, size: 12 * u, color: C.dim, alpha: hud });
      y += L.compact ? 66 * u : 84 * u;

      // counters
      const cw = (w - 16 * u) / 3;
      const counters = [['TOOL CALLS', cur.tools, C.cyan], ['EDITS', cur.edits, C.lime], ['PATCHES LIVE', cur.applied.length, C.amber]];
      counters.forEach(([lab, val, col], i) => {
        const cx0 = x + i * (cw + 8 * u);
        roundRect(ctx, cx0, y, cw, (L.compact ? 64 : 78) * u, 8 * u);
        ctx.fillStyle = rgba(col, 0.06 * hud); ctx.fill(); ctx.strokeStyle = rgba(col, 0.25 * hud); ctx.lineWidth = 1 * u; ctx.stroke();
        text(ctx, String(val), cx0 + cw / 2, y + (L.compact ? 36 : 44) * u, { font: DISPLAY, size: (L.compact ? 26 : 32) * u, weight: 700, align: 'center', color: col, glow: 10 * u, alpha: hud });
        text(ctx, lab, cx0 + cw / 2, y + (L.compact ? 54 : 66) * u, { font: MONO, size: 10 * u, align: 'center', color: C.muted, spacing: 1.5 * u, alpha: hud });
      });
      y += (L.compact ? 84 : 106) * u;

      // practice score sparkline
      const scores = cur.scores;
      text(ctx, 'PRACTICE SCORE', x, y, { font: MONO, size: 12 * u, color: C.muted, spacing: 2 * u, alpha: hud });
      const lastScore = scores.length ? scores[scores.length - 1].score : null;
      let shownScore = lastScore;
      if (b && b.type === 'practice' && b.score != null) shownScore = lerp(scores.length > 1 ? scores[scores.length - 2].score : 0, b.score, easeOut(seg(p, 0.35, 0.8)));
      text(ctx, shownScore == null ? '—' : fmt(shownScore), x + w, y + 4 * u, { font: DISPLAY, size: 26 * u, weight: 700, align: 'right', color: scoreColor(shownScore), glow: 12 * u, alpha: hud });
      y += 14 * u;
      const gh = L.compact ? r.y + r.h - y - 18 * u : 110 * u;
      if (gh < 30 * u) return;
      ctx.strokeStyle = 'rgba(120,200,255,0.10)'; ctx.lineWidth = 1 * u;
      for (let k = 0; k <= 4; k++) { ctx.beginPath(); ctx.moveTo(x, y + gh * k / 4); ctx.lineTo(x + w, y + gh * k / 4); ctx.stroke(); }
      if (scores.length) {
        const n = Math.max(scores.length, 6);
        const pts = scores.map((s, i) => [x + (i + 0.5) * w / n, y + gh * (1 - s.score)]);
        const grad = ctx.createLinearGradient(0, y, 0, y + gh);
        grad.addColorStop(0, 'rgba(163,230,53,0.30)'); grad.addColorStop(1, 'rgba(163,230,53,0)');
        ctx.beginPath(); ctx.moveTo(pts[0][0], y + gh); pts.forEach(q => ctx.lineTo(q[0], q[1])); ctx.lineTo(pts[pts.length - 1][0], y + gh); ctx.closePath();
        ctx.fillStyle = grad; ctx.globalAlpha = hud; ctx.fill(); ctx.globalAlpha = 1;
        ctx.beginPath(); pts.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]));
        ctx.strokeStyle = rgba(C.lime, hud); ctx.lineWidth = 2.5 * u; ctx.shadowColor = C.lime; ctx.shadowBlur = 10 * u; ctx.stroke(); ctx.shadowBlur = 0;
        pts.forEach((q, i) => { ctx.fillStyle = rgba(scoreColor(scores[i].score), hud); ctx.beginPath(); ctx.arc(q[0], q[1], 4.5 * u, 0, Math.PI * 2); ctx.fill(); });
      } else {
        text(ctx, tl.run.access === 'weights+activations+outputs' ? 'no practice runs yet' : 'no output access in this challenge', x + w / 2, y + gh / 2 + 4 * u, { font: MONO, size: 12 * u, align: 'center', color: C.dim, alpha: hud });
      }
      y += gh + 30 * u;
      if (L.compact || y > r.y + r.h - 60 * u) return;

      // patch stack
      text(ctx, 'PATCH STACK', x, y, { font: MONO, size: 12 * u, color: C.muted, spacing: 2 * u, alpha: hud });
      y += 14 * u;
      const maxRows = Math.floor((r.y + r.h - y - 16 * u) / (34 * u));
      const stack = cur.applied.slice(-maxRows);
      if (!stack.length) text(ctx, 'base model (unmodified)', x, y + 22 * u, { font: MONO, size: 13 * u, color: C.dim, alpha: hud });
      stack.forEach((pid, i) => {
        const yy = y + i * 34 * u;
        roundRect(ctx, x, yy, w, 28 * u, 6 * u);
        ctx.fillStyle = rgba(C.lime, 0.07 * hud); ctx.fill(); ctx.strokeStyle = rgba(C.lime, 0.35 * hud); ctx.stroke();
        text(ctx, '▲ ' + pid, x + 10 * u, yy + 19 * u, { font: MONO, size: 13 * u, color: C.lime, alpha: hud });
        const tt = targetText(cur.patches[pid]);
        text(ctx, tt.length > 26 ? tt.slice(0, 25) + '…' : tt, x + w - 10 * u, yy + 19 * u, { font: MONO, size: 11 * u, color: C.muted, align: 'right', alpha: hud });
      });
    }
    function idxSeed(b) { return Math.floor((b.t || 0) * 10); }

    // ---------------------------------------------------------------- tool log
    function drawLog(ctx, S) {
      const { L, u, hud, cur, time } = S, r = L.log;
      panel(ctx, r, 'TOOL CALL STREAM', L, hud, C.sky);
      const rows = Math.max(1, Math.floor((r.h - 56 * u) / (24 * u)));
      const lines = cur.log.slice(-rows);
      const x = r.x + 22 * u, y0 = r.y + 62 * u;
      const size = 14 * u;
      lines.forEach((ln, i) => {
        const y = y0 + i * 24 * u;
        const newest = i === lines.length - 1;
        const age = time - ln.t0;
        const a = hud * (newest ? 1 : 0.45 + 0.4 * (i / lines.length));
        text(ctx, clock(ln.rt || 0), x, y, { font: MONO, size: 12 * u, color: C.dim, alpha: a });
        text(ctx, (ln.ok ? '▸ ' : '✕ ') + ln.tool, x + 80 * u, y, { font: MONO, size, weight: 700, color: ln.ok ? ln.color : C.red, alpha: a });
        const tw = measure(ctx, '▸ ' + ln.tool, { font: MONO, size, weight: 700 });
        let d = ln.detail || '';
        const maxChars = Math.floor((r.w - 130 * u - tw) / (size * 0.6));
        if (d.length > maxChars) d = d.slice(0, Math.max(0, maxChars - 1)) + '…';
        if (newest) d = d.slice(0, Math.floor(age * 90));
        text(ctx, d, x + 92 * u + tw, y, { font: MONO, size: 13 * u, color: C.muted, alpha: a });
        if (newest && Math.floor(time * 3) % 2 === 0) {
          const dw = measure(ctx, d, { font: MONO, size: 13 * u });
          ctx.fillStyle = rgba(C.cyan, hud); ctx.fillRect(x + 94 * u + tw + dw, y - 12 * u, 8 * u, 15 * u);
        }
      });
      if (!lines.length) text(ctx, '> waiting for first tool call', x, y0, { font: MONO, size, color: C.dim, alpha: hud });
    }

    // ---------------------------------------------------------------- stage
    function drawStage(ctx, S) {
      const { L, u, hud } = S, st = L.stage;
      ctx.save();
      ctx.globalAlpha = hud;
      // stage frame: corner brackets only
      ctx.strokeStyle = 'rgba(120,200,255,0.30)'; ctx.lineWidth = 1.5 * u;
      const c = 22 * u;
      [[st.x, st.y, 1, 1], [st.x + st.w, st.y, -1, 1], [st.x, st.y + st.h, 1, -1], [st.x + st.w, st.y + st.h, -1, -1]].forEach(([x, y, dx, dy]) => {
        ctx.beginPath(); ctx.moveTo(x + dx * c, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * c); ctx.stroke();
      });
      ctx.restore();
      const kind = tl.arch.kind;
      ctx.save(); ctx.globalAlpha = hud;
      if (kind === 'transformer') drawTransformerStage(ctx, S);
      else if (kind === 'mlp') drawMLPStage(ctx, S);
      else drawTilesStage(ctx, S);
      drawStageOverlays(ctx, S);
      ctx.restore();
      // legend + honesty line
      drawLegend(ctx, S);
    }

    function drawLegend(ctx, S) {
      const { L, u, hud } = S, st = L.stage;
      const init = tl.initial || {};
      const items = [];
      if (init.damaged) items.push([C.red, 'damaged'], [C.magenta, 'sign-flipped'], [C.lime, 'patched'], [C.amber, 'changed'], [C.cyan, 'healthy']);
      else if (init.roles) items.push([C.violet, 'induction head'], [C.sky, 'previous-token head'], [C.dim, 'other']);
      else if (tl.arch.kind === 'mlp') items.push([C.red, 'hidden trigger input'], [C.cyan, 'task input'], [C.amber, 'edited']);
      else if (tl.arch.kind === 'transformer') items.push([C.cyan, 'head'], [C.amber, 'edited']);
      let x = st.x + 12 * u; const y = st.y + st.h - 14 * u;
      items.forEach(([col, lab]) => {
        ctx.fillStyle = rgba(col, hud); ctx.beginPath(); ctx.arc(x + 5 * u, y - 4 * u, 4.5 * u, 0, Math.PI * 2); ctx.fill();
        text(ctx, lab, x + 14 * u, y, { font: MONO, size: 11 * u, color: C.muted, alpha: hud });
        x += measure(ctx, lab, { font: MONO, size: 11 * u }) + 32 * u;
      });
      if (init.damaged || init.roles || init.planted || tl.arch.kind === 'mlp') {
        text(ctx, '◉ GROUND TRUTH SHOWN TO VIEWERS ONLY · THE AGENT CANNOT SEE IT', st.x + st.w - 12 * u, st.y + 22 * u,
             { font: MONO, size: 10 * u, color: C.muted, align: 'right', spacing: 1.2 * u, alpha: hud * 0.85 });
      }
    }

    // 3D projection for the transformer tower
    function makeProj(S, cxOff = 0, scale = 1) {
      const st = S.L.stage, t = S.time;
      const theta = 0.62 + 0.28 * Math.sin(t * 0.16);
      const R = Math.min(st.w * 0.24, st.h * 0.33) * scale;
      const cx = st.x + st.w * (tl.arch.dual ? 0.5 : S.L.land ? 0.57 : 0.6) + cxOff;
      const tokenBand = 96 * S.u;
      const top = st.y + st.h * 0.25, bottom = st.y + st.h - tokenBand - R * 0.3;
      const levelH = (bottom - top) / 3.15;
      const c = Math.cos(theta), s = Math.sin(theta);
      return (x, y, z) => {
        const xr = x * c - z * s, zr = x * s + z * c;
        const pz = 4.6 / (4.6 - zr);
        return { x: cx + xr * R * pz, y: bottom - y * levelH + zr * R * 0.24 * pz, p: pz, z: zr, R, cx };
      };
    }

    function headPos(l, h) { const a = Math.PI / 4 + h * Math.PI / 2 - (1 - l) * Math.PI / 4; return [0.72 * Math.cos(a), l + 1, 0.72 * Math.sin(a)]; }

    function ellipse(ctx, P, y, r, segs = 64) {
      ctx.beginPath();
      for (let i = 0; i <= segs; i++) { const a = i / segs * Math.PI * 2; const q = P(r * Math.cos(a), y, r * Math.sin(a)); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); }
    }

    function forwardWave(S) {
      const b = S.b; if (!b) return null;
      if (b.type === 'forward' || b.type === 'practice') return lerp(-0.6, 3.6, easeInOut(seg(S.p, 0.12, 0.8)));
      if (b.type === 'attention') return lerp(-0.6, 3.6, easeInOut(seg(S.p, 0.0, 0.5)));
      return null;
    }

    function drawTransformerStage(ctx, S) {
      const dual = tl.arch.dual;
      if (dual) {
        const off = S.L.stage.w * 0.24;
        drawTower(ctx, S, makeProj(S, -off, 0.72), { ghost: true, label: 'REFERENCE' });
        drawTower(ctx, S, makeProj(S, off, 0.72), { label: 'SUSPECT (fine-tuned)' });
        if (S.b && S.b.type === 'diff') drawDiffLinks(ctx, S, makeProj(S, -off, 0.72), makeProj(S, off, 0.72));
      } else {
        drawTower(ctx, S, makeProj(S), {});
      }
      drawTokenStrip(ctx, S);
    }

    function drawTower(ctx, S, P, o) {
      const { u, time } = S;
      const ga = o.ghost ? 0.45 : 1;
      const wave = forwardWave(S);
      const st = S.L.stage;
      const after = S.b && (S.b.type === 'edit' || S.b.type === 'revert') && S.p < (S.b.type === 'edit' ? 0.62 : 0.5) ? S.prev : S.cur;
      const init = tl.initial || {};
      // platforms
      const levels = [
        { y: 0, name: 'EMBED', short: 'EMBED', color: C.sky },
        { y: 1, name: 'LAYER 0 · ATTENTION', short: 'L0', color: C.cyan },
        { y: 2, name: 'LAYER 1 · ATTENTION', short: 'L1', color: C.cyan },
        { y: 3, name: 'UNEMBED · 32 TOKENS', short: 'OUT', color: C.violet },
      ];
      const nodes = [];
      for (let l = 0; l < 2; l++) for (let h = 0; h < 4; h++) {
        const [x, y, z] = headPos(l, h); const q = P(x, y, z); nodes.push({ l, h, q, key: headKey(l, h) });
      }
      const beamTop = P(0, 3.35, 0), beamBot = P(0, -0.5, 0);
      levels.forEach(lv => {
        const hot = wave != null ? Math.exp(-Math.pow((wave - lv.y) * 2.2, 2)) : 0;
        const touched = (lv.y === 0 && partTouched('embed', after)) || (lv.y === 3 && partTouched('unembed', after));
        const col = touched ? C.amber : lv.color;
        ellipse(ctx, P, lv.y, 1.08);
        const c0 = P(0, lv.y, 0);
        const g = ctx.createRadialGradient(c0.x, c0.y, 0, c0.x, c0.y, c0.R * 1.1);
        g.addColorStop(0, rgba(col, (0.10 + 0.25 * hot) * ga)); g.addColorStop(1, rgba(col, 0.02 * ga));
        ctx.fillStyle = g; ctx.fill();
        ctx.strokeStyle = rgba(col, (0.45 + 0.5 * hot) * ga); ctx.lineWidth = (1.4 + 2 * hot) * u;
        ctx.shadowColor = col; ctx.shadowBlur = (6 + 18 * hot) * u; ctx.stroke(); ctx.shadowBlur = 0;
        ctx.setLineDash([4 * u, 7 * u]); ellipse(ctx, P, lv.y, 0.45); ctx.strokeStyle = rgba(col, 0.25 * ga); ctx.lineWidth = 1 * u; ctx.stroke(); ctx.setLineDash([]);
        // rotating ticks
        for (let k = 0; k < 24; k++) {
          const a = k / 24 * Math.PI * 2 + time * 0.15 * (lv.y % 2 ? -1 : 1);
          const q1 = P(1.08 * Math.cos(a), lv.y, 1.08 * Math.sin(a)), q2 = P(1.16 * Math.cos(a), lv.y, 1.16 * Math.sin(a));
          ctx.strokeStyle = rgba(col, (k % 6 === 0 ? 0.7 : 0.25) * ga); ctx.lineWidth = 1.2 * u;
          ctx.beginPath(); ctx.moveTo(q1.x, q1.y); ctx.lineTo(q2.x, q2.y); ctx.stroke();
        }
        {
          text(ctx, S.L.compact ? lv.short : lv.name, c0.x - c0.R * 1.22 * c0.p - 8 * u, c0.y + 4 * u,
               { font: MONO, size: 11 * u, color: rgba(col, 1), spacing: 1.5 * u, align: 'right', alpha: (0.55 + 0.45 * hot) * ga * S.hud });
        }
        if (lv.y === 3) drawVocabRing(ctx, S, P, after, ga);
      });
      // beam (behind heads with z<0 drawn first)
      const back = nodes.filter(n => n.q.z < 0), front = nodes.filter(n => n.q.z >= 0);
      back.forEach(n => drawHead(ctx, S, P, n, after, init, ga, wave));
      drawBeam(ctx, S, beamBot, beamTop, wave, ga);
      front.forEach(n => drawHead(ctx, S, P, n, after, init, ga, wave));
      if (o.label) {
        const q = P(0, 3.6, 0);
        text(ctx, o.label, q.x, q.y - 6 * u, { font: MONO, size: 13 * u, weight: 700, color: o.ghost ? C.muted : C.magenta, align: 'center', spacing: 2 * u, alpha: S.hud });
        if (!o.ghost && init.planted) {
          text(ctx, `planted (hidden): after ${init.planted.trigger} → predict ${init.planted.forced}`, q.x, q.y + 12 * u,
               { font: MONO, size: 11 * u, color: C.red, align: 'center', alpha: S.hud * 0.9 });
        }
      }
      // store for overlays
      S.towerNodes = S.towerNodes || []; S.towerNodes.push({ P, nodes, ghost: !!o.ghost });
      if (!o.ghost) S.P = P;
    }

    function drawVocabRing(ctx, S, P, state, ga) {
      const { u } = S;
      const init = tl.initial || {};
      const goal = init.goal || {};
      const special = {};
      if (goal.target) special[goal.target] = [C.red, 'BAN ' + goal.target];
      if (goal.trigger) special[goal.trigger] = [C.sky, 'TRIGGER ' + goal.trigger];
      if (goal.output) special[goal.output] = [C.lime, 'OUTPUT ' + goal.output];
      const ub = partTouched('unembed', state);
      for (let k = 0; k < 32; k++) {
        const a = k / 32 * Math.PI * 2 - Math.PI / 2;
        const q = P(0.86 * Math.cos(a), 3, 0.86 * Math.sin(a));
        const name = 't' + k;
        const sp = special[name];
        let col = sp ? sp[0] : C.violet;
        if (ub && ub.row === k) col = C.amber;
        const r = (sp ? 4.5 : 2.6) * u * q.p;
        ctx.fillStyle = rgba(col, (sp ? 1 : 0.55) * ga);
        if (sp) { ctx.shadowColor = col; ctx.shadowBlur = 12 * u; }
        ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
        if (sp) text(ctx, sp[1], q.x, q.y - 10 * u, { font: MONO, size: 11 * u, weight: 700, color: col, align: 'center', alpha: S.hud * ga });
        if (sp && goal.target === name && ub && ub.row === k) {
          ctx.strokeStyle = rgba(C.red, ga); ctx.lineWidth = 2 * u;
          ctx.beginPath(); ctx.moveTo(q.x - 7 * u, q.y - 7 * u); ctx.lineTo(q.x + 7 * u, q.y + 7 * u); ctx.moveTo(q.x + 7 * u, q.y - 7 * u); ctx.lineTo(q.x - 7 * u, q.y + 7 * u); ctx.stroke();
        }
      }
      if (goal.trigger && goal.output) {
        const ka = Number(goal.trigger.slice(1)), kb = Number(goal.output.slice(1));
        const qa = P(0.86 * Math.cos(ka / 32 * Math.PI * 2 - Math.PI / 2), 3, 0.86 * Math.sin(ka / 32 * Math.PI * 2 - Math.PI / 2));
        const qb = P(0.86 * Math.cos(kb / 32 * Math.PI * 2 - Math.PI / 2), 3, 0.86 * Math.sin(kb / 32 * Math.PI * 2 - Math.PI / 2));
        const c = P(0, 3.15, 0);
        ctx.strokeStyle = rgba(ub ? C.lime : C.dim, 0.8 * ga); ctx.lineWidth = (ub ? 2.5 : 1.2) * u;
        ctx.setLineDash(ub ? [] : [5 * u, 5 * u]);
        ctx.beginPath(); ctx.moveTo(qa.x, qa.y); ctx.quadraticCurveTo(c.x, c.y - 40 * u, qb.x, qb.y); ctx.stroke(); ctx.setLineDash([]);
      }
    }

    function drawBeam(ctx, S, a, b, wave, ga) {
      const { u, time } = S;
      const g = ctx.createLinearGradient(0, a.y, 0, b.y);
      g.addColorStop(0, rgba(C.sky, 0.15 * ga)); g.addColorStop(0.5, rgba(C.cyan, 0.65 * ga)); g.addColorStop(1, rgba(C.violet, 0.55 * ga));
      ctx.strokeStyle = g; ctx.lineWidth = 5 * u; ctx.shadowColor = C.cyan; ctx.shadowBlur = 18 * u;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.lineWidth = 1.6 * u; ctx.strokeStyle = rgba(C.white, 0.75 * ga); ctx.stroke(); ctx.shadowBlur = 0;
      // flowing particles
      const n = wave != null ? 46 : 26;
      for (let i = 0; i < n; i++) {
        const k = (time * (0.22 + 0.12 * hash(i)) + hash(i + 11)) % 1;
        const x = lerp(a.x, b.x, k) + Math.sin(time * 3 + i) * 3 * u, y = lerp(a.y, b.y, k);
        ctx.fillStyle = rgba(i % 4 ? C.cyan : C.white, (0.4 + 0.6 * hash(i + 4)) * ga);
        ctx.fillRect(x - 1.5 * u, y - 1.5 * u, 3 * u, 3 * u);
      }
      if (wave != null) {
        const wy = lerp(a.y, b.y, (wave + 0.5) / 3.85);
        const rg = ctx.createRadialGradient(a.x, wy, 0, a.x, wy, 60 * u);
        rg.addColorStop(0, rgba(C.white, 0.9 * ga)); rg.addColorStop(0.3, rgba(C.cyan, 0.5 * ga)); rg.addColorStop(1, rgba(C.cyan, 0));
        ctx.fillStyle = rg; ctx.fillRect(a.x - 60 * u, wy - 60 * u, 120 * u, 120 * u);
      }
    }

    function drawHead(ctx, S, P, n, state, init, ga, wave) {
      const { u, time, frame } = S;
      let status = init.roles ? (init.roles[n.key] || 'none') : headStatus(n.key, state, init);
      const meta = STATUS[status] || STATUS.ok;
      const q = n.q, sz = 24 * u * q.p;
      const base = P(0, n.l + 1, 0);
      // link to beam
      ctx.strokeStyle = rgba(meta.color, 0.35 * ga); ctx.lineWidth = 1.4 * u;
      ctx.beginPath(); ctx.moveTo(base.x, base.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      for (let i = 0; i < 2; i++) {
        const k = (time * 0.6 + i * 0.5 + hash(n.l * 4 + n.h)) % 1;
        ctx.fillStyle = rgba(meta.color, 0.9 * ga);
        ctx.fillRect(lerp(base.x, q.x, k) - 1.5 * u, lerp(base.y, q.y, k) - 1.5 * u, 3 * u, 3 * u);
      }
      const hot = wave != null ? Math.exp(-Math.pow((wave - (n.l + 1)) * 2.5, 2)) : 0;
      let flick = 1;
      if (status === 'damaged' || status === 'flip') flick = 0.55 + 0.45 * (hash(frame * 0.5 + n.h * 7 + n.l * 3) > 0.25 ? 1 : 0.2);
      const glitchX = (status === 'damaged' || status === 'flip') && hash(frame + n.h * 13) > 0.85 ? (hash(frame + 2) - 0.5) * 10 * u : 0;
      const x = q.x + glitchX, y = q.y;
      // pillar
      const hgt = 26 * u * q.p;
      ctx.save();
      ctx.globalAlpha = ga * flick;
      const col = mix(meta.color, C.white, hot * 0.7);
      ctx.shadowColor = meta.color; ctx.shadowBlur = (14 + 26 * hot) * u;
      hexPath(ctx, x, y - hgt, sz, 0.55);
      const g = ctx.createLinearGradient(x, y - hgt - sz, x, y);
      g.addColorStop(0, rgba(col, 0.95)); g.addColorStop(1, rgba(meta.color, 0.35));
      ctx.fillStyle = g; ctx.fill();
      ctx.strokeStyle = rgba(C.white, 0.85); ctx.lineWidth = 1.3 * u; ctx.stroke();
      ctx.shadowBlur = 0;
      // sides
      for (let i = 0; i < 6; i++) {
        const a = Math.PI / 3 * i;
        if (Math.sin(a) < -0.1) continue;
        const px = x + Math.cos(a) * sz, py = y - hgt + Math.sin(a) * sz * 0.55;
        ctx.strokeStyle = rgba(meta.color, 0.7); ctx.lineWidth = 1.2 * u;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py + hgt); ctx.stroke();
      }
      hexPath(ctx, x, y, sz, 0.55); ctx.strokeStyle = rgba(meta.color, 0.5); ctx.stroke();
      ctx.restore();
      // damage cracks
      if (status === 'damaged' || status === 'flip') {
        ctx.strokeStyle = rgba(C.white, 0.7 * ga * flick); ctx.lineWidth = 1.2 * u;
        ctx.beginPath(); ctx.moveTo(x - sz * 0.5, y - hgt - sz * 0.1); ctx.lineTo(x - sz * 0.1, y - hgt + sz * 0.15); ctx.lineTo(x + sz * 0.15, y - hgt - sz * 0.2); ctx.lineTo(x + sz * 0.55, y - hgt + sz * 0.05); ctx.stroke();
      }
      // label
      text(ctx, n.key, x, y - hgt - sz * 0.75, { font: MONO, size: 13 * u * Math.min(1.1, q.p), weight: 700, color: C.text, align: 'center', alpha: ga * S.hud });
      if (meta.label) text(ctx, meta.label + (status === 'flip' ? ' ⇅' : ''), x, y + sz * 0.95 + 6 * u, { font: MONO, size: 10 * u, weight: 700, color: meta.color, align: 'center', spacing: 1.2 * u, alpha: ga * S.hud * (status === 'damaged' || status === 'flip' ? flick : 1) });
    }

    function hexPath(ctx, x, y, r, squash) {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) { const a = Math.PI / 3 * i; const px = x + Math.cos(a) * r, py = y + Math.sin(a) * r * squash; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.closePath();
    }

    function drawDiffLinks(ctx, S, PA, PB) {
      const { u, p, time } = S;
      for (let l = 0; l < 2; l++) for (let h = 0; h < 4; h++) {
        const [x, y, z] = headPos(l, h); const a = PA(x, y, z), b = PB(x, y, z);
        const k = easeOut(seg(p, 0.05 + 0.06 * (l * 4 + h), 0.5 + 0.06 * (l * 4 + h)));
        const heat = hash(l * 4 + h + 3);
        ctx.strokeStyle = rgba(heat > 0.6 ? C.magenta : C.cyan, 0.65 * (1 - seg(p, 0.85, 1)));
        ctx.lineWidth = (1 + 2.5 * heat) * u;
        ctx.beginPath(); ctx.moveTo(a.x, a.y - 20 * u); ctx.lineTo(lerp(a.x, b.x, k), lerp(a.y, b.y, k) - 20 * u); ctx.stroke();
      }
      // vocab-ring diff (the planted change lives in embeddings/unembedding)
      const a = PA(0, 3, 0), b = PB(0, 3, 0);
      ctx.strokeStyle = rgba(C.red, 0.9 * (0.5 + 0.5 * Math.sin(time * 10))); ctx.lineWidth = 3 * u;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(lerp(a.x, b.x, easeOut(seg(p, 0.2, 0.7))), lerp(a.y, b.y, easeOut(seg(p, 0.2, 0.7)))); ctx.stroke();
      text(ctx, 'Δ WEIGHTS', (a.x + b.x) / 2, a.y - 30 * u, { font: DISPLAY, size: 20 * u, weight: 700, color: C.magenta, align: 'center', glow: 14 * u, alpha: seg(p, 0.1, 0.3) * (1 - seg(p, 0.85, 1)) });
    }

    function drawTokenStrip(ctx, S) {
      const { L, u, b, p, cur, time } = S, st = L.stage;
      const tokens = (b && b.tokens && b.tokens.length ? b.tokens : cur.lastTokens) || defaultTokens();
      const n = Math.min(tokens.length, 24);
      const y = st.y + st.h - 62 * u;
      const cw = Math.min(46 * u, (st.w - 40 * u) / n), x0 = st.x + st.w / 2 - cw * n / 2;
      const fwd = b && (b.type === 'forward' || b.type === 'practice' || b.type === 'attention');
      const lit = fwd ? seg(p, 0, 0.25) * n : -1;
      for (let i = 0; i < n; i++) {
        const x = x0 + i * cw;
        const on = i < lit;
        roundRect(ctx, x + 2 * u, y, cw - 4 * u, 26 * u, 5 * u);
        ctx.fillStyle = on ? rgba(C.sky, 0.35) : 'rgba(20,30,50,0.65)'; ctx.fill();
        ctx.strokeStyle = on ? rgba(C.sky, 0.9) : 'rgba(120,200,255,0.18)'; ctx.lineWidth = 1 * u; ctx.stroke();
        const tok = typeof tokens[i] === 'number' ? 't' + tokens[i] : String(tokens[i]).slice(0, 4);
        text(ctx, tok, x + cw / 2, y + 17.5 * u, { font: MONO, size: Math.min(12 * u, cw * 0.3), color: on ? C.white : C.muted, align: 'center' });
      }
      text(ctx, 'INPUT TOKENS', x0 + 2 * u, y - 8 * u, { font: MONO, size: 10 * u, color: C.dim, spacing: 1.5 * u });
      // feed lines into the tower base
      if (fwd && S.P) {
        const base = S.P(0, -0.5, 0);
        for (let i = 0; i < n; i += 2) {
          const k = seg(p, 0.08 + i * 0.006, 0.35 + i * 0.006);
          if (k <= 0 || k >= 1) continue;
          const sx = x0 + i * cw + cw / 2;
          ctx.strokeStyle = rgba(C.sky, 0.7 * (1 - k)); ctx.lineWidth = 1.5 * u;
          ctx.beginPath(); ctx.moveTo(sx, y); ctx.quadraticCurveTo(sx, base.y + 20 * u, lerp(sx, base.x, k), lerp(y, base.y, k)); ctx.stroke();
        }
      }
      // attention arcs
      if (b && b.type === 'attention') {
        const layers = b.layers || [1];
        const heads = [C.violet, C.cyan, C.magenta, C.lime];
        const arcs = [];
        for (let q = 1; q < n; q++) {
          if (layers.includes(1)) {
            let prevIdx = -1;
            for (let k = q - 1; k >= 0; k--) if (tokens[k] === tokens[q]) { prevIdx = k; break; }
            if (prevIdx >= 0 && prevIdx + 1 < q) arcs.push([q, prevIdx + 1, 1]);
          }
          if (layers.includes(0) && q % 3 === 0) arcs.push([q, q - 1, 0]);
        }
        arcs.forEach((a, i) => {
          const k = easeOut(seg(p, 0.25 + i * 0.03, 0.55 + i * 0.03));
          if (k <= 0) return;
          const xa = x0 + a[0] * cw + cw / 2, xb = x0 + a[1] * cw + cw / 2;
          const hgt = Math.min(150 * u, Math.abs(xa - xb) * 0.32 + 16 * u);
          const col = heads[(a[0] + a[2]) % 4];
          ctx.strokeStyle = rgba(col, 0.85 * (1 - seg(p, 0.9, 1))); ctx.lineWidth = 2 * u; ctx.shadowColor = col; ctx.shadowBlur = 8 * u;
          ctx.beginPath();
          const steps = 24;
          for (let j = 0; j <= Math.floor(steps * k); j++) {
            const s = j / steps; const xx = lerp(xa, xb, s); const yy = y - 4 * u - Math.sin(s * Math.PI) * hgt;
            j ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy);
          }
          ctx.stroke(); ctx.shadowBlur = 0;
        });
      }
    }
    function defaultTokens() { const a = []; for (let i = 0; i < 24; i++) a.push(Math.floor(hash(i + 50) * 32)); return a; }

    // ---------------------------------------------------------------- MLP stage (backdoor-removal)
    function mlpGeom(S) {
      const st = S.L.stage, u = S.u, a = tl.arch;
      const cols = [a.d_in, a.hidden, a.d_out];
      const xs = [st.x + st.w * 0.18, st.x + st.w * 0.5, st.x + st.w * 0.82];
      const top = st.y + 70 * u, bot = st.y + st.h - 70 * u;
      return cols.map((n, ci) => Array.from({ length: n }, (_, i) => ({ x: xs[ci], y: n === 1 ? (top + bot) / 2 : lerp(top, bot, (i + 0.5) / n), i, ci })));
    }
    function mlpCut(state) {
      const cut = new Set(); let l0 = false, l1 = false;
      for (const pid of state.applied) for (const t of (state.patches[pid] || [])) {
        if (t.kind !== 'mlp') continue;
        if (t.layer === 0) { if (t.inputs) t.inputs.forEach(i => cut.add(i)); else l0 = true; } else l1 = true;
      }
      return { cut, l0, l1 };
    }
    function drawMLPStage(ctx, S) {
      const { u, time, b, p } = S;
      const G = mlpGeom(S);
      const init = tl.initial || {};
      const trig = new Set(init.trigger_inputs || []), task = new Set(init.task_inputs || []);
      const state = b && (b.type === 'edit' || b.type === 'revert') && p < 0.6 ? S.prev : S.cur;
      const { cut, l0, l1 } = mlpCut(state);
      const fwd = b && (b.type === 'forward' || b.type === 'practice') ? easeInOut(seg(p, 0.1, 0.85)) : null;
      // edges
      for (let c = 0; c < 2; c++) for (const a of G[c]) for (const z of G[c + 1]) {
        const w = hash(a.i * 31 + z.i * 7 + c * 101);
        const isCut = c === 0 && cut.has(a.i);
        const col = c === 0 && trig.has(a.i) ? C.red : c === 0 && task.has(a.i) ? C.cyan : C.violet;
        ctx.strokeStyle = rgba(isCut ? C.dim : col, isCut ? 0.08 : 0.05 + 0.18 * w);
        ctx.lineWidth = (0.6 + 1.4 * w) * u;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(z.x, z.y); ctx.stroke();
        if (fwd != null && !isCut && w > 0.55) {
          const k = clamp(fwd * 2 - c);
          if (k > 0 && k < 1) { ctx.fillStyle = rgba(col, 0.9); ctx.fillRect(lerp(a.x, z.x, k) - 2 * u, lerp(a.y, z.y, k) - 2 * u, 4 * u, 4 * u); }
        }
      }
      // nodes
      G.forEach((col, ci) => col.forEach(nd => {
        let color = ci === 0 ? (trig.has(nd.i) ? C.red : task.has(nd.i) ? C.cyan : C.dim) : ci === 1 ? C.violet : C.lime;
        if ((ci === 1 && l0) || (ci === 2 && l1)) color = C.amber;
        const r = (ci === 1 ? 9 : 13) * u;
        const pulse = ci === 0 && trig.has(nd.i) && !cut.has(nd.i) ? 0.5 + 0.5 * Math.sin(time * 5 + nd.i) : 0;
        ctx.shadowColor = color; ctx.shadowBlur = (10 + 14 * pulse) * u;
        ctx.fillStyle = rgba(color, cut.has(nd.i) && ci === 0 ? 0.3 : 0.9);
        ctx.beginPath(); ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
        if (ci === 0) {
          if (trig.has(nd.i) && !cut.has(nd.i)) { ctx.strokeStyle = rgba(C.red, 0.4 + 0.6 * pulse); ctx.lineWidth = 2 * u; ctx.beginPath(); ctx.arc(nd.x, nd.y, r + 6 * u + 6 * u * pulse, 0, Math.PI * 2); ctx.stroke(); }
          text(ctx, 'x' + nd.i, nd.x - r - 10 * u, nd.y + 5 * u, { font: MONO, size: 13 * u, color: C.muted, align: 'right' });
          const tag = cut.has(nd.i) ? 'CUT' : trig.has(nd.i) ? 'TRIGGER' : task.has(nd.i) ? 'task' : '';
          if (tag) text(ctx, tag, nd.x - r - 46 * u, nd.y + 5 * u, { font: MONO, size: 11 * u, weight: 700, color: cut.has(nd.i) ? C.amber : trig.has(nd.i) ? C.red : C.cyan, align: 'right', spacing: 1 * u });
        }
        if (ci === 2) text(ctx, 'class ' + nd.i, nd.x + r + 12 * u, nd.y + 5 * u, { font: MONO, size: 13 * u, color: C.muted });
      }));
      const lab = ['INPUTS', 'HIDDEN · ' + tl.arch.hidden + ' RELU', 'OUTPUTS'];
      G.forEach((col, ci) => text(ctx, lab[ci], col[0].x, S.L.stage.y + 40 * u, { font: MONO, size: 12 * u, color: C.muted, align: 'center', spacing: 2 * u }));
      S.mlp = G;
    }

    // ---------------------------------------------------------------- tiles stage (zoo-sprint, name-that-function)
    function tileRects(S) {
      const st = S.L.stage, u = S.u, n = tl.arch.tiles.length;
      const cols = S.L.land ? 4 : 4, rows = Math.ceil(n / cols);
      const gap = 16 * u, top = st.y + 24 * u, avail = st.h - 70 * u;
      const tw = (st.w - gap * (cols + 1)) / cols, th = Math.min((avail - gap * (rows - 1)) / rows, tw * 0.9);
      const y0 = top + (avail - (th * rows + gap * (rows - 1))) / 2;
      return tl.arch.tiles.map((tile, i) => ({ x: st.x + gap + (i % cols) * (tw + gap), y: y0 + Math.floor(i / cols) * (th + gap), w: tw, h: th, tile, i }));
    }
    function drawTilesStage(ctx, S) {
      const { u, b, p, time, cur } = S;
      const rects = tileRects(S);
      rects.forEach(r => {
        const active = b && b.tile === r.i && b.type !== 'note';
        const info = cur.tiles[r.i] || {};
        const col = active ? (b.type === 'error' ? C.red : C.cyan) : info.touched ? C.violet : C.dim;
        chamfer(ctx, r, 12 * u);
        ctx.fillStyle = active ? rgba(col, 0.10) : 'rgba(12,18,34,0.7)'; ctx.fill();
        ctx.strokeStyle = rgba(col, active ? 0.95 : 0.4); ctx.lineWidth = (active ? 2.2 : 1.2) * u;
        if (active) { ctx.shadowColor = col; ctx.shadowBlur = 18 * u; }
        ctx.stroke(); ctx.shadowBlur = 0;
        drawFamilyGlyph(ctx, r, r.tile.family, col, time, u, active);
        text(ctx, String(r.tile.family || '').toUpperCase().replace('_', ' '), r.x + 12 * u, r.y + 22 * u, { font: MONO, size: 11 * u, weight: 700, color: col, spacing: 1.2 * u });
        text(ctx, r.tile.net || '', r.x + r.w - 10 * u, r.y + r.h - 10 * u, { font: MONO, size: 10 * u, color: C.dim, align: 'right' });
        if (info.touched && !active) text(ctx, info.errors ? '✕ ' + info.errors : '✓', r.x + r.w - 12 * u, r.y + 22 * u, { font: MONO, size: 13 * u, weight: 700, color: info.errors ? C.red : C.lime, align: 'right' });
        if (active && b.type === 'scan') {
          const sy = r.y + ((p * 2) % 1) * r.h;
          const g = ctx.createLinearGradient(0, sy - 16 * u, 0, sy + 2 * u);
          g.addColorStop(0, rgba(C.cyan, 0)); g.addColorStop(1, rgba(C.cyan, 0.8));
          ctx.fillStyle = g; ctx.fillRect(r.x + 2 * u, sy - 16 * u, r.w - 4 * u, 18 * u);
        }
      });
    }
    function drawFamilyGlyph(ctx, r, fam, col, time, u, active) {
      const cx = r.x + r.w / 2, cy = r.y + r.h / 2 + 4 * u, s = Math.min(r.w, r.h) * 0.3;
      ctx.save(); ctx.strokeStyle = rgba(col, active ? 0.95 : 0.55); ctx.fillStyle = rgba(col, active ? 0.95 : 0.55); ctx.lineWidth = 1.3 * u;
      if (/cnn/.test(fam)) {
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { const k = 0.3 + 0.7 * hash(i * 4 + j + Math.floor(time * 2)); ctx.globalAlpha = active ? k : 0.5; ctx.fillRect(cx - s + i * s * 0.5, cy - s + j * s * 0.5, s * 0.42, s * 0.42); }
      } else if (/tf/.test(fam)) {
        for (let l = 0; l < 3; l++) { ctx.beginPath(); ctx.ellipse(cx, cy + s * 0.7 - l * s * 0.7, s, s * 0.28, 0, 0, Math.PI * 2); ctx.stroke(); }
        ctx.beginPath(); ctx.moveTo(cx, cy + s); ctx.lineTo(cx, cy - s); ctx.stroke();
      } else if (/rl/.test(fam)) {
        ctx.beginPath(); ctx.arc(cx, cy, s * 0.8, 0, Math.PI * 2); ctx.stroke();
        const a = time * 1.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * s, cy + Math.sin(a) * s); ctx.stroke();
        ctx.beginPath(); ctx.arc(cx + Math.cos(a) * s, cy + Math.sin(a) * s, 4 * u, 0, Math.PI * 2); ctx.fill();
      } else {
        const L = [4, 6, 2]; const pts = L.map((n, ci) => Array.from({ length: n }, (_, i) => [cx - s + ci * s, cy - s * 0.8 + (i + 0.5) * (1.6 * s / n)]));
        ctx.globalAlpha = 0.35;
        for (let c = 0; c < 2; c++) for (const a of pts[c]) for (const z of pts[c + 1]) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(z[0], z[1]); ctx.stroke(); }
        ctx.globalAlpha = 1;
        pts.flat().forEach(q => { ctx.beginPath(); ctx.arc(q[0], q[1], 3.2 * u, 0, Math.PI * 2); ctx.fill(); });
      }
      ctx.restore();
    }

    // ---------------------------------------------------------------- per-beat stage effects
    function targetPoints(S, targets) {
      const pts = [];
      if (tl.arch.kind === 'transformer' && S.P) {
        for (const t of targets || []) {
          if (t.kind === 'heads') {
            const hs = t.heads || [0, 1, 2, 3];
            hs.forEach(h => { const [x, y, z] = headPos(t.layer, h); const q = S.P(x, y, z); pts.push({ x: q.x, y: q.y - 26 * S.u * q.p, r: 34 * S.u * q.p }); });
          } else if (t.kind === 'unembed' || t.kind === 'embed') {
            const lvl = t.kind === 'unembed' ? 3 : 0;
            if (t.row != null) { const a = t.row / 32 * Math.PI * 2 - Math.PI / 2; const q = S.P(0.86 * Math.cos(a), lvl, 0.86 * Math.sin(a)); pts.push({ x: q.x, y: q.y, r: 22 * S.u }); }
            else { const q = S.P(0, lvl, 0); pts.push({ x: q.x, y: q.y, r: q.R * 0.9, wide: true }); }
          } else { const q = S.P(0, 1.5, 0); pts.push({ x: q.x, y: q.y, r: q.R, wide: true }); }
        }
      } else if (tl.arch.kind === 'mlp' && S.mlp) {
        for (const t of targets || []) {
          if (t.kind === 'mlp' && t.layer === 0 && t.inputs) t.inputs.forEach(i => { const n = S.mlp[0][i]; if (n) pts.push({ x: n.x, y: n.y, r: 26 * S.u }); });
          else { const col = S.mlp[t.layer === 1 ? 2 : 1]; const mid = col[Math.floor(col.length / 2)]; pts.push({ x: mid.x, y: mid.y, r: 60 * S.u, wide: true }); }
        }
      } else if (tl.arch.kind === 'tiles') {
        const r = tileRects(S)[S.b.tile || 0]; if (r) pts.push({ x: r.x + r.w / 2, y: r.y + r.h / 2, r: Math.min(r.w, r.h) * 0.45 });
      }
      return pts;
    }

    function reticle(ctx, x, y, r, col, u, alpha, rot = 0) {
      ctx.save(); ctx.translate(x, y); ctx.rotate(rot);
      ctx.strokeStyle = rgba(col, alpha); ctx.lineWidth = 2 * u; ctx.shadowColor = col; ctx.shadowBlur = 10 * u;
      const c = r * 0.45;
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([dx, dy]) => {
        ctx.beginPath(); ctx.moveTo(dx * r, dy * (r - c)); ctx.lineTo(dx * r, dy * r); ctx.lineTo(dx * (r - c), dy * r); ctx.stroke();
      });
      ctx.restore();
    }

    function drawStageOverlays(ctx, S) {
      const { b, p, u, L, time, lt } = S, st = L.stage;
      if (!b) return;
      if (b.type === 'scan') {
        let pts = [];
        const tg = b.target || {};
        if (tl.arch.kind === 'transformer' && S.P) {
          if (tg.kind === 'heads') pts = targetPoints(S, [tg]);
          else pts = targetPoints(S, [{ kind: tg.kind === 'embed' || tg.kind === 'unembed' ? tg.kind : 'all' }]);
        } else if (tl.arch.kind === 'mlp' && S.mlp) pts = targetPoints(S, [tg.kind === 'mlp' ? tg : { kind: 'mlp', layer: 0 }]);
        pts.forEach((q, i) => {
          const k = easeOut(seg(p, 0, 0.3));
          reticle(ctx, q.x, q.y, q.r * lerp(2.4, 1, k), C.cyan, u, k * (1 - seg(p, 0.9, 1)), (1 - k) * 1.2);
        });
        // laser sweep across the stage
        if (pts.length) {
          const ys = pts.map(q => q.y), ymin = Math.min(...ys) - 50 * u, ymax = Math.max(...ys) + 30 * u;
          const sy = lerp(ymin, ymax, easeInOut(seg(p, 0.15, 0.75)));
          const g = ctx.createLinearGradient(0, sy - 30 * u, 0, sy);
          g.addColorStop(0, rgba(C.cyan, 0)); g.addColorStop(1, rgba(C.cyan, 0.35 * (1 - seg(p, 0.75, 0.9))));
          ctx.fillStyle = g; ctx.fillRect(st.x, sy - 30 * u, st.w, 30 * u);
          ctx.fillStyle = rgba(C.white, 0.9 * (1 - seg(p, 0.75, 0.9))); ctx.fillRect(st.x, sy, st.w, 1.5 * u);
          // hex readout
          const hx = Math.max(...pts.map(q => q.x)) + 50 * u;
          for (let i = 0; i < 6; i++) {
            const v = Math.floor(hash(i + Math.floor(time * 12) * 7) * 0xffff).toString(16).padStart(4, '0');
            text(ctx, '0x' + v + '  ' + (hash(i + Math.floor(time * 8)) * 2 - 1).toFixed(3), Math.min(hx, st.x + st.w - 150 * u), ymin + 20 * u + i * 17 * u,
                 { font: MONO, size: 12 * u, color: C.cyan, alpha: 0.7 * seg(p, 0.15, 0.3) * (1 - seg(p, 0.85, 1)) });
          }
        }
        banner(ctx, S, '◎ ' + (b.tool || 'scan').toUpperCase().replace(/_/g, ' '), b.label || '', C.cyan);
      } else if (b.type === 'forward' || b.type === 'practice') {
        banner(ctx, S, b.type === 'practice' ? (b.split === 'heldout' ? '▶ HELD-OUT EVALUATION' : '▶ PRACTICE RUN') : '▶ FORWARD PASS', (b.tool || '').replace(/_/g, ' '), C.sky);
        if (b.type === 'practice' && b.score != null) {
          const k = easeBack(seg(p, 0.55, 0.75));
          const val = lerp(0, b.score, easeOut(seg(p, 0.55, 0.85)));
          const x = st.x + st.w - 30 * u, y = st.y + 130 * u;
          ctx.save(); ctx.globalAlpha = k * (1 - seg(p, 0.94, 1));
          text(ctx, 'SCORE', x, y - 58 * u, { font: MONO, size: 13 * u, color: C.muted, align: 'right', spacing: 3 * u });
          text(ctx, fmt(val), x, y, { font: DISPLAY, size: 64 * u * lerp(0.6, 1, k), weight: 800, color: scoreColor(b.score), align: 'right', glow: 24 * u });
          ctx.restore();
        }
      } else if (b.type === 'attention') {
        banner(ctx, S, '⌁ ATTENTION PROBE', 'layers ' + (b.layers || []).join(', '), C.violet);
      } else if (b.type === 'diff') {
        banner(ctx, S, '⇄ WEIGHT DIFF', b.label || '', C.magenta);
      } else if (b.type === 'edit') {
        drawSurgery(ctx, S);
      } else if (b.type === 'revert') {
        const pts = targetPoints(S, b.targets);
        pts.forEach(q => {
          const k = seg(p, 0.1, 0.7);
          ctx.strokeStyle = rgba(C.amber, 0.9 * (1 - k)); ctx.lineWidth = 3 * u;
          ctx.beginPath(); ctx.ellipse(q.x, q.y, q.r * lerp(3, 0.4, k), q.r * lerp(3, 0.4, k) * 0.55, 0, 0, Math.PI * 2); ctx.stroke();
        });
        banner(ctx, S, '⟲ REVERT', (b.patch_id || '') + ' · bit-exact', C.amber);
      } else if (b.type === 'note') {
        // a soft "thinking" pulse over the stage
        const k = 0.5 + 0.5 * Math.sin(time * 4);
        ctx.strokeStyle = rgba(b.src === 'thinking' ? C.violet : C.cyan, 0.12 + 0.1 * k); ctx.lineWidth = 2 * u;
        chamfer(ctx, { x: st.x + 6 * u, y: st.y + 6 * u, w: st.w - 12 * u, h: st.h - 12 * u }, 18 * u); ctx.stroke();
      } else if (b.type === 'generic') {
        banner(ctx, S, '· ' + (b.tool || '').toUpperCase().replace(/_/g, ' '), '', C.muted);
      } else if (b.type === 'error') {
        banner(ctx, S, '✕ ' + (b.tool || '').toUpperCase().replace(/_/g, ' ') + ' REFUSED', b.text || '', C.red);
      }
    }

    function banner(ctx, S, title, sub, col) {
      const { L, u, p, lt, b } = S, st = L.stage;
      const k = easeOut(seg(lt, 0, 0.25)) * (1 - seg(p, 0.88, 1));
      if (k <= 0) return;
      const x = st.x + 24 * u, y = st.y + 44 * u;
      const shown = title.slice(0, Math.ceil(lt * 60));
      text(ctx, shown, x + (1 - k) * -30 * u, y, { font: DISPLAY, size: 24 * u, weight: 700, color: col, glow: 16 * u, spacing: 2 * u, alpha: k });
      if (sub) {
        let s2 = sub; const maxW = st.w * 0.6;
        while (measure(ctx, s2, { font: MONO, size: 13 * u }) > maxW && s2.length > 4) s2 = s2.slice(0, -2);
        if (s2 !== sub) s2 += '…';
        text(ctx, s2, x, y + 24 * u, { font: MONO, size: 13 * u, color: C.muted, alpha: k });
      }
    }

    function drawSurgery(ctx, S) {
      const { b, p, u, L, time, lt } = S, st = L.stage;
      const pts = targetPoints(S, b.targets);
      const reap = b.reapply;
      // phase A: target lock
      const kA = easeOut(seg(p, 0, 0.22));
      pts.forEach((q, i) => reticle(ctx, q.x, q.y, q.r * lerp(2.6, 1.05, kA), reap ? C.lime : C.red, u, kA * (1 - seg(p, 0.92, 1)), (1 - kA) * 1.6 + (p > 0.22 ? Math.sin(time * 6) * 0.05 : 0)));
      if (p < 0.3 && !reap) {
        text(ctx, 'TARGET LOCK · ' + pts.length + (pts.length === 1 ? ' SITE' : ' SITES'), st.x + st.w / 2, st.y + 84 * u,
             { font: MONO, size: 15 * u, weight: 700, color: C.red, align: 'center', spacing: 3 * u, alpha: seg(p, 0.03, 0.08) * (Math.floor(time * 8) % 2 ? 1 : 0.5) });
      }
      // phase B: laser surgery
      const kB = seg(p, 0.22, 0.62);
      if (kB > 0 && kB < 1) {
        const ox = st.x + st.w / 2, oy = st.y - 10 * u;
        pts.forEach((q, i) => {
          const jitter = (hash(S.frame + i) - 0.5) * 6 * u;
          ctx.strokeStyle = rgba(C.white, 0.95); ctx.lineWidth = (3 + 2 * Math.sin(time * 40 + i)) * u; ctx.shadowColor = C.lime; ctx.shadowBlur = 26 * u;
          ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(q.x + jitter, q.y); ctx.stroke();
          ctx.strokeStyle = rgba(C.lime, 0.6); ctx.lineWidth = 9 * u; ctx.stroke(); ctx.shadowBlur = 0;
          // sparks
          for (let s = 0; s < 22; s++) {
            const a = hash(s * 3 + i * 17) * Math.PI * 2, sp = (40 + 90 * hash(s + i * 5)) * u;
            const life = (kB * 3 + hash(s + i)) % 1;
            ctx.fillStyle = rgba(s % 3 ? C.lime : C.white, 1 - life);
            ctx.fillRect(q.x + Math.cos(a) * sp * life, q.y + Math.sin(a) * sp * life + 30 * u * life * life, 3 * u, 3 * u);
          }
          const flare = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, 50 * u);
          flare.addColorStop(0, 'rgba(255,255,255,0.95)'); flare.addColorStop(0.4, rgba(C.lime, 0.4)); flare.addColorStop(1, rgba(C.lime, 0));
          ctx.fillStyle = flare; ctx.fillRect(q.x - 50 * u, q.y - 50 * u, 100 * u, 100 * u);
        });
      }
      // phase C: shockwave + stamp
      const kC = seg(p, 0.6, 0.95);
      if (kC > 0) {
        pts.forEach(q => {
          ctx.strokeStyle = rgba(C.lime, 0.9 * (1 - kC)); ctx.lineWidth = 3 * u * (1 - kC) + 1 * u;
          ctx.beginPath(); ctx.ellipse(q.x, q.y, q.r * (1 + 4 * kC), q.r * (1 + 4 * kC) * 0.55, 0, 0, Math.PI * 2); ctx.stroke();
        });
        if (p > 0.6 && p < 0.66) { ctx.fillStyle = `rgba(200,255,150,${0.18 * (1 - seg(p, 0.6, 0.66))})`; ctx.fillRect(0, 0, L.W, L.H); }
      }
      const kS = easeBack(seg(p, 0.62, 0.74)) * (1 - seg(p, 0.95, 1));
      if (kS > 0) {
        const x = st.x + 24 * u, y = st.y + 54 * u;
        ctx.save(); ctx.globalAlpha = clamp(kS);
        text(ctx, reap ? 'PATCH RE-APPLIED' : 'PATCH APPLIED', x, y, { font: DISPLAY, size: (L.compact ? 24 : 30) * u * lerp(1.4, 1, clamp(kS)), weight: 800, color: C.lime, glow: 20 * u, spacing: 2 * u });
        const bits = [b.patch_id, ...(b.ops || []).map(o => o.replace(/_/g, ' ')), b.params ? b.params.toLocaleString('en-US') + ' params' : null].filter(Boolean).join('  ·  ');
        text(ctx, bits, x, y + 26 * u, { font: MONO, size: 13 * u, color: C.muted });
        if (b.test && b.test.before != null && b.test.after != null) {
          const yy = y + 70 * u;
          text(ctx, 'REGRESSION CHECK', x, yy - 4 * u, { font: MONO, size: 11 * u, color: C.muted, spacing: 2 * u });
          const s1 = fmt(b.test.before), s2 = p < 0.68 ? '·'.repeat(1 + Math.floor(time * 6) % 3) : fmt(lerp(b.test.before, b.test.after, easeOut(seg(p, 0.68, 0.88))));
          text(ctx, s1, x, yy + 34 * u, { font: DISPLAY, size: 30 * u, weight: 700, color: scoreColor(b.test.before) });
          const w1 = measure(ctx, s1, { font: DISPLAY, size: 30 * u, weight: 700 });
          text(ctx, '→', x + w1 + 14 * u, yy + 32 * u, { font: SANS, size: 28 * u, weight: 700, color: C.text });
          text(ctx, s2, x + w1 + 52 * u, yy + 34 * u, { font: DISPLAY, size: 30 * u, weight: 700, color: p < 0.68 ? C.cyan : scoreColor(lerp(b.test.before, b.test.after, easeOut(seg(p, 0.68, 0.88)))), glow: 16 * u });
        }
        ctx.restore();
      }
    }

    // ---------------------------------------------------------------- full-screen moments
    function drawIntro(ctx, S) {
      const { L, u, lt, time } = S, W = L.W, H = L.H;
      // CRT power-on line
      const k0 = seg(lt, 0, 0.5);
      if (k0 < 1) {
        ctx.fillStyle = `rgba(0,0,0,${1 - k0 * 0.3})`; ctx.fillRect(0, 0, W, H);
        const lw = W * easeOut(k0); ctx.fillStyle = rgba(C.cyan, 0.9); ctx.shadowColor = C.cyan; ctx.shadowBlur = 30 * u;
        ctx.fillRect(W / 2 - lw / 2, H / 2 - 1.5 * u, lw, 3 * u); ctx.shadowBlur = 0;
      }
      const out = 1 - seg(lt, 3.4, 4.3);
      const kT = seg(lt, 0.5, 1.4);
      if (kT > 0 && out > 0) {
        const glitch = (1 - kT) * 18 * u + (hash(Math.floor(time * 20)) > 0.9 ? 6 * u : 0);
        const size = (L.land ? 120 : 96) * u;
        const y = H * 0.42;
        ctx.save(); ctx.globalAlpha = out;
        ctx.globalCompositeOperation = 'lighter';
        text(ctx, 'WSP ARENA', W / 2 - glitch, y, { font: DISPLAY, size, weight: 800, color: C.red, align: 'center', spacing: 10 * u, alpha: kT * 0.7 * out });
        text(ctx, 'WSP ARENA', W / 2 + glitch, y, { font: DISPLAY, size, weight: 800, color: C.cyan, align: 'center', spacing: 10 * u, alpha: kT * 0.7 * out });
        ctx.globalCompositeOperation = 'source-over';
        text(ctx, 'WSP ARENA', W / 2, y, { font: DISPLAY, size, weight: 800, color: C.white, align: 'center', spacing: 10 * u, glow: 30 * u, glowColor: C.cyan, alpha: kT * out });
        const sub = 'WEIGHT-SPACE PROGRAMMING · AGENT BENCHMARK';
        text(ctx, sub.slice(0, Math.floor(seg(lt, 1.0, 1.8) * sub.length)), W / 2, y + 56 * u, { font: MONO, size: 20 * u, color: C.cyan, align: 'center', spacing: 5 * u, alpha: out });
        // challenge card
        const kc = easeOut(seg(lt, 1.8, 2.4));
        if (kc > 0) {
          const run = tl.run;
          const cardW = Math.min(W - 80 * u, 980 * u), cardH = 150 * u;
          const r = { x: W / 2 - cardW / 2, y: y + 96 * u + (1 - kc) * 30 * u, w: cardW, h: cardH };
          ctx.globalAlpha = kc * out;
          chamfer(ctx, r, 18 * u); ctx.fillStyle = 'rgba(10,16,32,0.85)'; ctx.fill(); ctx.strokeStyle = rgba(C.cyan, 0.5); ctx.lineWidth = 1.5 * u; ctx.stroke();
          text(ctx, 'CHALLENGE', r.x + 28 * u, r.y + 36 * u, { font: MONO, size: 13 * u, color: C.muted, spacing: 3 * u, alpha: kc * out });
          text(ctx, run.title.toUpperCase(), r.x + 28 * u, r.y + 78 * u, { font: DISPLAY, size: 34 * u, weight: 700, color: C.text, spacing: 2 * u, alpha: kc * out });
          text(ctx, 'AGENT  ' + String(run.model).toUpperCase(), r.x + r.w - 28 * u, r.y + 36 * u, { font: MONO, size: 15 * u, weight: 700, color: C.cyan, align: 'right', spacing: 2 * u, alpha: kc * out });
          text(ctx, `SEED ${run.seed} · ${run.access.toUpperCase()} · ${run.budget_per_session ? run.budget_per_session + ' PASSES' : 'NO FORWARD PASSES'}`, r.x + 28 * u, r.y + 118 * u,
               { font: MONO, size: 14 * u, color: C.muted, spacing: 1.5 * u, alpha: kc * out });
        }
        const ke = seg(lt, 2.9, 3.2) * (1 - seg(lt, 3.6, 3.9));
        if (ke > 0) text(ctx, '— ENGAGE —', W / 2, H * 0.86, { font: DISPLAY, size: 26 * u, weight: 700, color: C.lime, align: 'center', spacing: 8 * u, glow: 20 * u, alpha: ke * (Math.floor(time * 10) % 2 ? 1 : 0.6) });
        ctx.restore();
      }
    }

    function drawErrorFlash(ctx, S) {
      const { L, lt, u } = S;
      const k = 1 - seg(lt, 0, 0.6);
      if (k <= 0) return;
      const v = ctx.createRadialGradient(L.W / 2, L.H / 2, Math.min(L.W, L.H) * 0.3, L.W / 2, L.H / 2, Math.max(L.W, L.H) * 0.7);
      v.addColorStop(0, 'rgba(244,63,94,0)'); v.addColorStop(1, `rgba(244,63,94,${0.45 * k})`);
      ctx.fillStyle = v; ctx.fillRect(0, 0, L.W, L.H);
    }

    function drawRewind(ctx, S) {
      const { L, p, u, frame } = S;
      const k = Math.sin(clamp(p / 0.7) * Math.PI);
      if (k <= 0.01) return;
      for (let i = 0; i < 14; i++) {
        const y = hash(frame * 3 + i) * L.H, h = (2 + 10 * hash(frame + i * 7)) * u;
        ctx.fillStyle = rgba(i % 2 ? C.amber : C.cyan, 0.10 * k);
        ctx.fillRect((hash(frame + i) - 0.5) * 60 * u, y, L.W, h);
      }
      text(ctx, '◀◀', L.stage.x + L.stage.w - 30 * u, L.stage.y + 60 * u, { font: DISPLAY, size: 40 * u, weight: 800, color: C.amber, align: 'right', glow: 20 * u, alpha: k });
    }

    function drawAwaiting(ctx, S) {
      const { L, u, time } = S, st = L.stage;
      const k = 0.5 + 0.5 * Math.sin(time * 3);
      text(ctx, 'AWAITING AGENT' + '.'.repeat(1 + Math.floor(time * 2) % 3), st.x + st.w / 2, st.y + 48 * u, { font: DISPLAY, size: 20 * u, weight: 700, color: C.cyan, align: 'center', spacing: 4 * u, alpha: 0.5 + 0.5 * k });
    }

    function drawSubmit(ctx, S) {
      const { L, u, lt, time, outro } = S, W = L.W, H = L.H, run = tl.run;
      const score = run.score == null ? 0 : run.score;
      const kD = easeOut(seg(lt, 0, 0.6));
      ctx.fillStyle = `rgba(2,4,10,${(outro ? 0.9 : 0.84) * kD})`; ctx.fillRect(0, 0, W, H);
      if (outro) return;
      const cx = W / 2, cy = H * (L.land ? 0.45 : 0.4), R = Math.min(W, H) * 0.2;
      // banner
      const kB = seg(lt, 0.2, 0.5) * (1 - seg(lt, 1.2, 1.5));
      if (kB > 0) text(ctx, 'ANSWER SUBMITTED', cx, cy, { font: DISPLAY, size: 44 * u, weight: 800, color: C.white, align: 'center', spacing: 6 * u, glow: 24 * u, alpha: kB });
      const kR = seg(lt, 1.3, 1.6);
      if (kR <= 0) return;
      // ring gauge
      ctx.save(); ctx.globalAlpha = kR;
      ctx.lineWidth = 16 * u; ctx.strokeStyle = 'rgba(120,200,255,0.12)';
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
      for (let i = 0; i < 60; i++) {
        const a = i / 60 * Math.PI * 2 - Math.PI / 2;
        ctx.strokeStyle = rgba(C.cyan, i % 5 ? 0.18 : 0.5); ctx.lineWidth = 2 * u;
        ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * (R + 18 * u), cy + Math.sin(a) * (R + 18 * u)); ctx.lineTo(cx + Math.cos(a) * (R + (i % 5 ? 24 : 30) * u), cy + Math.sin(a) * (R + (i % 5 ? 24 : 30) * u)); ctx.stroke();
      }
      const kF = easeInOut(seg(lt, 1.5, 3.0));
      const val = score * kF;
      const col = scoreColor(kF >= 1 ? score : val);
      ctx.lineWidth = 16 * u; ctx.lineCap = 'round'; ctx.strokeStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 30 * u;
      ctx.beginPath(); ctx.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.max(0.001, val)); ctx.stroke();
      ctx.shadowBlur = 0; ctx.lineCap = 'butt';
      // spinning outer arcs
      ctx.strokeStyle = rgba(C.violet, 0.6); ctx.lineWidth = 2 * u;
      ctx.beginPath(); ctx.arc(cx, cy, R + 44 * u, time * 0.8, time * 0.8 + 1.2); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, cy, R + 44 * u, time * 0.8 + Math.PI, time * 0.8 + Math.PI + 0.6); ctx.stroke();
      text(ctx, fmt(val), cx, cy + 22 * u, { font: DISPLAY, size: R * 0.5, weight: 800, color: C.white, align: 'center', glow: 26 * u, glowColor: col });
      text(ctx, 'SCORE', cx, cy - R * 0.36, { font: MONO, size: 15 * u, color: C.muted, align: 'center', spacing: 5 * u });
      ctx.restore();
      // verdict
      const kV = easeBack(seg(lt, 3.0, 3.35));
      if (kV > 0) {
        const verdict = score >= 0.9 ? 'SOLVED' : score >= 0.6 ? 'STRONG PARTIAL' : score > 0.2 ? 'PARTIAL' : score > 0 ? 'BARELY' : 'FAILED';
        ctx.save(); ctx.globalAlpha = clamp(kV);
        text(ctx, verdict, cx, cy + R + 92 * u, { font: DISPLAY, size: 52 * u * lerp(1.6, 1, clamp(kV)), weight: 800, color: scoreColor(score), align: 'center', spacing: 8 * u, glow: 28 * u });
        ctx.restore();
        // detail metrics
        const d = run.detail || {};
        const items = flattenDetail(d).slice(0, 4);
        items.forEach((it, i) => {
          const ki = seg(lt, 3.5 + i * 0.25, 3.8 + i * 0.25);
          text(ctx, it, cx, cy + R + 140 * u + i * 26 * u, { font: MONO, size: 16 * u, color: C.muted, align: 'center', alpha: ki });
        });
        if (score >= 0.9) confetti(ctx, S, cx, cy, seg(lt, 3.0, 6.5));
      }
      // label for honesty
      text(ctx, `${String(run.model)} · ${run.title} · seed ${run.seed}`, cx, H - 40 * u, { font: MONO, size: 13 * u, color: C.dim, align: 'center', alpha: kR });
    }

    function flattenDetail(d, prefix = '') {
      const out = [];
      for (const [k, v] of Object.entries(d || {})) {
        if (k === 'reference') continue;
        if (v && typeof v === 'object' && !Array.isArray(v)) out.push(...flattenDetail(v, k + ' · '));
        else if (typeof v === 'number') out.push(`${prefix}${k.replace(/_/g, ' ')}  ${Number.isInteger(v) && !/accura|rate|score|jaccard|success/.test(k) ? v : v.toFixed(3)}`);
        else if (typeof v === 'boolean') out.push(`${prefix}${k.replace(/_/g, ' ')}  ${v ? '✓' : '✕'}`);
        else if (typeof v === 'string' && v.length < 24) out.push(`${prefix}${k.replace(/_/g, ' ')}  ${v}`);
      }
      return out;
    }

    function confetti(ctx, S, cx, cy, k) {
      if (k <= 0 || k >= 1) return;
      const u = S.u, cols = [C.lime, C.cyan, C.violet, C.magenta, C.amber, C.white];
      for (let i = 0; i < 140; i++) {
        const a = hash(i * 1.3) * Math.PI * 2, sp = (300 + 700 * hash(i + 2)) * u;
        const tt = k * 3.5;
        const x = cx + Math.cos(a) * sp * tt * 0.6, y = cy + Math.sin(a) * sp * tt * 0.6 + 260 * u * tt * tt;
        ctx.save(); ctx.translate(x, y); ctx.rotate(tt * 8 * (hash(i + 5) - 0.5));
        ctx.fillStyle = rgba(cols[i % cols.length], 1 - k);
        ctx.fillRect(-4 * u, -2 * u, 8 * u, 4 * u); ctx.restore();
      }
    }

    function drawOutro(ctx, S) {
      const { L, u, lt, time } = S, W = L.W, H = L.H, run = tl.run;
      const k = easeOut(seg(lt, 0.1, 0.9));
      const y = H * 0.36;
      ctx.save(); ctx.globalAlpha = k;
      text(ctx, 'WSP BENCH', W / 2, y, { font: DISPLAY, size: (L.land ? 96 : 80) * u, weight: 800, color: C.white, align: 'center', spacing: 12 * u, glow: 30 * u, glowColor: C.cyan });
      const sc = run.score == null ? '' : fmt(run.score);
      const scripted = /scripted/i.test(run.model || '');
      const line = scripted ? `The no-AI baseline scored ${sc}. Can your model beat it?` : `${run.model} scored ${sc}. Can your model beat it?`;
      text(ctx, line, W / 2, y + 70 * u, { font: SANS, size: 30 * u, weight: 600, color: C.text, align: 'center' });
      const k2 = seg(lt, 0.8, 1.4);
      const cmd = '$ wsp bench run --model <your-model>';
      const cw = measure(ctx, cmd, { font: MONO, size: 24 * u }) + 60 * u;
      const r = { x: W / 2 - cw / 2, y: y + 120 * u, w: cw, h: 62 * u };
      ctx.globalAlpha = k2;
      chamfer(ctx, r, 12 * u); ctx.fillStyle = 'rgba(10,16,32,0.9)'; ctx.fill(); ctx.strokeStyle = rgba(C.lime, 0.6); ctx.lineWidth = 1.5 * u; ctx.stroke();
      text(ctx, cmd.slice(0, Math.floor(seg(lt, 0.9, 2.0) * cmd.length)), r.x + 30 * u, r.y + 40 * u, { font: MONO, size: 24 * u, color: C.lime, glow: 10 * u });
      ctx.globalAlpha = seg(lt, 1.8, 2.4);
      text(ctx, tl.site || '', W / 2, r.y + r.h + 70 * u, { font: MONO, size: 24 * u, weight: 700, color: C.cyan, align: 'center', spacing: 1 * u, glow: 14 * u });
      text(ctx, 'open leaderboard · 9 challenges · every run re-scored from its seed', W / 2, r.y + r.h + 108 * u, { font: MONO, size: 16 * u, color: C.muted, align: 'center' });
      ctx.restore();
    }

    // ---------------------------------------------------------------- playback
    function tick(now) {
      if (!playing) return;
      if (last == null) last = now;
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      let sp = speed;
      if (live && !tl.run.finished && tl.duration - t > 12) sp = Math.max(speed, 2.5);
      t += dt * sp;
      if (!live || tl.run.finished) { if (t >= tl.duration) { t = tl.duration - 0.001; playing = false; } }
      render(t);
      if (onTick) onTick(t);
      if (playing) raf = requestAnimationFrame(tick);
    }

    return {
      render: (time) => { t = time; render(time); },
      play() { if (playing) return; if (t >= tl.duration - 0.01 && (!live || tl.run.finished)) t = 0; playing = true; last = null; raf = requestAnimationFrame(tick); },
      pause() { playing = false; if (raf) cancelAnimationFrame(raf); },
      seek(time) { t = clamp(time, 0, tl.duration); render(t); if (onTick) onTick(t); },
      setSpeed(s) { speed = s; },
      setTimeline(next) { tl = next; states = precompute(tl); },
      setLive(v) { live = v; },
      onTick(fn) { onTick = fn; },
      get time() { return t; },
      get duration() { return tl.duration; },
      get playing() { return playing; },
      get timeline() { return tl; },
    };
  }

  async function ready() {
    if (!document.fonts) return;
    try {
      await Promise.all(['800 40px Orbitron', '600 40px Orbitron', '400 20px "JetBrains Mono"', '700 20px "JetBrains Mono"']
        .map(f => document.fonts.load(f)));
    } catch (e) { /* fall back to system fonts */ }
  }

  global.WSPArena = { create, ready, layout, precompute };
})(typeof window !== 'undefined' ? window : globalThis);
