/* ═══════════════════════════════════════════════════════════════
   MoE All-to-All Communication Visualization
   ES5 style, no modules, global namespace.
   Requires js/moe-engine.js loaded before this file.
   ═══════════════════════════════════════════════════════════════ */
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

/* ───────────────────────────────────────────────────────────────
   DOM refs
   ─────────────────────────────────────────────────────────────── */
var $themeToggle = document.getElementById('themeToggle');
var $gpuSeg = document.getElementById('gpuSeg');
var $expSeg = document.getElementById('expSeg');
var $tokPresets = document.getElementById('tokPresets');
var $tokInput = document.getElementById('tokInput');
var $topkSeg = document.getElementById('topkSeg');
var $skewSlider = document.getElementById('skewSlider');
var $skewVal = document.getElementById('skewVal');
var $scenarioRow = document.getElementById('scenarioRow');
var $tokSize = document.getElementById('tokSize');
var $bandwidth = document.getElementById('bandwidth');
var $expSpeed = document.getElementById('expSpeed');
var $phaseBar = document.getElementById('phaseBar');
var $playBtn = document.getElementById('playBtn');
var $stepBtn = document.getElementById('stepBtn');
var $resetBtn = document.getElementById('resetBtn');
var $rerouteBtn = document.getElementById('rerouteBtn');
var $speedSeg = document.getElementById('speedSeg');
var $scrubber = document.getElementById('scrubber');
var $scrubFrom = document.getElementById('scrubFrom');
var $scrubTo = document.getElementById('scrubTo');
var $vizScroll = document.getElementById('vizScroll');
var $vizSvg = document.getElementById('vizSvg');
var $vizLegend = document.getElementById('vizLegend');
var $imbalanceValue = document.getElementById('imbalanceValue');
var $imbalanceHint = document.getElementById('imbalanceHint');
var $metricList = document.getElementById('metricList');
var $hotCard = document.getElementById('hotCard');
var $affinityChart = document.getElementById('affinityChart');
var $affinityAxis = document.getElementById('affinityAxis');
var $affinitySection = document.getElementById('affinitySection');

/* ───────────────────────────────────────────────────────────────
   State
   ─────────────────────────────────────────────────────────────── */
var state = {
  numGpus: 4,
  expertsPerGpu: 2,
  numTokens: 8,
  topk: 2,
  skew: 0.3,
  seed: 0,
  tokenSizeKB: 8,
  bandwidthGBs: 300,
  expertSpeedK: 100,
  speed: 1,
  playing: false,
  started: false,
  waveElapsed: 0,
  route: null
};
state.seed = MoEEngine.generateSeed();

// Phase definitions: name + base duration (ms) at 1x.
var PHASES = [
  { key: 'dispatch', name: 'Dispatch', dur: 1000 },
  { key: 'send',     name: 'Send',     dur: 2000 },
  { key: 'compute',  name: 'Compute',  dur: 1500 },
  { key: 'combine',  name: 'Combine',  dur: 2000 }
];
var TOTAL_DUR = PHASES.reduce(function (s, p) { return s + p.dur; }, 0);

// Cached palette read from CSS variables; refreshed on theme change.
var palette = {};

/* ───────────────────────────────────────────────────────────────
   Theme + seg-control helpers (match calculator.js convention)
   ─────────────────────────────────────────────────────────────── */
function initTheme() {
  var stored = localStorage.getItem('kv-theme');
  if (stored === 'dark' || stored === 'light') {
    document.documentElement.setAttribute('data-theme', stored);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.setAttribute('data-theme', 'dark');
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
}
function toggleTheme() {
  var current = document.documentElement.getAttribute('data-theme');
  var next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('kv-theme', next);
}
function initSegControl(container, callback) {
  var btns = container.querySelectorAll('.seg-option');
  btns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      btns.forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      callback(btn.getAttribute('data-value'));
    });
  });
}
function getCSSVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

initTheme();
$themeToggle.addEventListener('click', function () {
  toggleTheme();
  refreshPalette();
  if (state.route) { renderVizStructure(); renderFrame(); }
});

function refreshPalette() {
  palette = {
    accent: getCSSVar('--accent') || '#6366f1',
    accent2: getCSSVar('--accent2') || '#818cf8',
    green: getCSSVar('--green') || '#22c55e',
    orange: getCSSVar('--orange') || '#f59e0b',
    orangeDark: getCSSVar('--orange-dark') || '#d97706',
    red: getCSSVar('--red') || '#ef4444',
    text2: getCSSVar('--text2') || '#475569',
    text3: getCSSVar('--text3') || '#94a3b8',
    surface2: getCSSVar('--surface2') || '#eef0f4',
    border: getCSSVar('--border') || '#e2e5eb',
    bg: getCSSVar('--bg') || '#ffffff'
  };
}
refreshPalette();

/* ───────────────────────────────────────────────────────────────
   Color helpers (interpolate using theme palette values)
   ─────────────────────────────────────────────────────────────── */
function hexToRgb(h) {
  h = h.replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  var n = parseInt(h, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}
function mixRgb(a, b, t) {
  return {
    r: Math.round(a.r + (b.r - a.r) * t),
    g: Math.round(a.g + (b.g - a.g) * t),
    b: Math.round(a.b + (b.b - a.b) * t)
  };
}
function rgbStr(c) { return 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')'; }

// loadColor: green(0) -> orange(0.5) -> red(1) by load/max.
// Overloaded experts (load > 1.5*avg) forced to red.
function loadColor(load, max, avg) {
  if (load <= 0) return palette.surface2;
  var green = hexToRgb(palette.green);
  var orange = hexToRgb(palette.orange);
  var red = hexToRgb(palette.red);
  if (avg > 0 && load > 1.5 * avg) return palette.red;
  var t = max > 0 ? load / max : 0;
  var c;
  if (t < 0.5) c = mixRgb(green, orange, t * 2);
  else c = mixRgb(orange, red, Math.min(1, (t - 0.5) * 2));
  return rgbStr(c);
}

// Source-GPU palette for particles: cycle through a set built from vars.
function srcColor(gpuIdx) {
  var set = [palette.accent, palette.green, palette.orange, palette.red, palette.accent2];
  return set[gpuIdx % set.length];
}

/* ───────────────────────────────────────────────────────────────
   Tooltip system
   ─────────────────────────────────────────────────────────────── */
var $tooltip = document.createElement('div');
$tooltip.className = 'moe-tooltip';
$tooltip.style.display = 'none';
document.body.appendChild($tooltip);

function showTooltip(html, el) {
  $tooltip.innerHTML = html;
  $tooltip.style.display = 'block';
  var rect = el.getBoundingClientRect();
  var tw = $tooltip.offsetWidth;
  var th = $tooltip.offsetHeight;
  var x = rect.left + rect.width / 2 - tw / 2;
  var y = rect.top - th - 8;
  if (y < 8) y = rect.bottom + 8;
  if (x < 8) x = 8;
  if (x + tw > window.innerWidth - 8) x = window.innerWidth - tw - 8;
  $tooltip.style.left = x + 'px';
  $tooltip.style.top = y + 'px';
}

function hideTooltip() {
  $tooltip.style.display = 'none';
}

function sendTooltipHtml(g) {
  var R = state.route;
  if (!R || !R.expertFlow) return '';
  var E = R.expertsPerGpu;
  var total = 0;
  var groups = [];
  for (var dg = 0; dg < R.numGpus; dg++) {
    var gpuCnt = R.dispatchMatrix[g][dg];
    if (gpuCnt <= 0) continue;
    total += gpuCnt;
    var exps = [];
    for (var e = 0; e < E; e++) {
      var cnt = R.expertFlow[g][dg][e];
      if (cnt > 0) exps.push({ exp: dg * E + e, cnt: cnt });
    }
    exps.sort(function(a, b) { return b.cnt - a.cnt; });
    groups.push({ dg: dg, gpuCnt: gpuCnt, exps: exps, local: dg === g });
  }
  groups.sort(function(a, b) { return b.gpuCnt - a.gpuCnt; });
  var html = '<div class="tt-head">GPU ' + g + ' \u00b7 \u53d1\u9001 ' + total + ' tokens</div>';
  groups.forEach(function(gr) {
    var local = gr.local ? ' <span class="tt-local">(\u672c\u673a)</span>' : '';
    html += '<div class="tt-row tt-subhead"><span>\u2192 GPU' + gr.dg + local + '</span><span class="tt-val">' + gr.gpuCnt + ' tok</span></div>';
    gr.exps.forEach(function(ex) {
      html += '<div class="tt-row tt-sub"><span class="tt-indent">e' + ex.exp + '</span><span class="tt-val">' + ex.cnt + '</span></div>';
    });
  });
  return html;
}

function expertTooltipHtml(g, e) {
  var R = state.route;
  if (!R || !R.expertFlow) return '';
  var E = R.expertsPerGpu;
  var expIdx = g * E + e;
  var total = R.expertLoads[g][e];
  var items = [];
  for (var sg = 0; sg < R.numGpus; sg++) {
    var cnt = R.expertFlow[sg][g][e];
    if (cnt > 0) items.push({ sg: sg, cnt: cnt });
  }
  items.sort(function(a, b) { return b.cnt - a.cnt; });
  var html = '<div class="tt-head">\u4e13\u5bb6 e' + expIdx + ' \u00b7 \u63a5\u6536 ' + total + ' tokens</div>';
  items.forEach(function(it) {
    var local = it.sg === g ? ' <span class="tt-local">(\u672c\u673a)</span>' : '';
    html += '<div class="tt-row"><span>\u2190 GPU' + it.sg + local + '</span><span class="tt-val">' + it.cnt + ' tok</span></div>';
  });
  return html;
}

function outputTooltipHtml(g) {
  var R = state.route;
  if (!R) return '';
  var total = 0;
  var items = [];
  for (var eg = 0; eg < R.numGpus; eg++) {
    var cnt = R.combineMatrix[eg][g];
    if (cnt > 0) {
      items.push({ eg: eg, cnt: cnt });
      total += cnt;
    }
  }
  items.sort(function(a, b) { return b.cnt - a.cnt; });
  var html = '<div class="tt-head">GPU ' + g + ' \u00b7 \u6536\u56de ' + total + ' tokens</div>';
  items.forEach(function(it) {
    var local = it.eg === g ? ' <span class="tt-local">(\u672c\u673a)</span>' : '';
    html += '<div class="tt-row"><span>\u2190 GPU' + it.eg + local + '</span><span class="tt-val">' + it.cnt + ' tok</span></div>';
  });
  return html;
}

/* ───────────────────────────────────────────────────────────────
   Control wiring
   ─────────────────────────────────────────────────────────────── */
initSegControl($gpuSeg, function (v) { state.numGpus = parseInt(v, 10); rerouteAndRender(); });
initSegControl($expSeg, function (v) { state.expertsPerGpu = parseInt(v, 10); rerouteAndRender(); });
initSegControl($topkSeg, function (v) { state.topk = parseInt(v, 10); rerouteAndRender(); });
initSegControl($speedSeg, function (v) { state.speed = parseFloat(v); });

$tokPresets.querySelectorAll('.preset-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    $tokPresets.querySelectorAll('.preset-btn').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    $tokInput.value = btn.getAttribute('data-value');
    state.numTokens = parseInt(btn.getAttribute('data-value'), 10);
    rerouteAndRender();
  });
});
$tokInput.addEventListener('input', function () {
  $tokPresets.querySelectorAll('.preset-btn').forEach(function (b) { b.classList.remove('active'); });
  var v = parseInt($tokInput.value, 10);
  if (v > 0) { state.numTokens = v; rerouteAndRender(); }
});

$skewSlider.addEventListener('input', function () {
  state.skew = parseFloat($skewSlider.value);
  $skewVal.textContent = state.skew.toFixed(2);
  syncScenarioActive();
});
// Re-route on release (perf) — input gives live slider readout.
$skewSlider.addEventListener('change', function () {
  rerouteAndRender();
});

$scenarioRow.querySelectorAll('.scenario-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var v = parseFloat(btn.getAttribute('data-value'));
    state.skew = v;
    $skewSlider.value = String(v);
    $skewVal.textContent = v.toFixed(2);
    syncScenarioActive();
    rerouteAndRender();
  });
});
function syncScenarioActive() {
  $scenarioRow.querySelectorAll('.scenario-btn').forEach(function (b) {
    var bv = parseFloat(b.getAttribute('data-value'));
    b.classList.toggle('active', Math.abs(bv - state.skew) < 0.001);
  });
}

[$tokSize, $bandwidth, $expSpeed].forEach(function (el) {
  el.addEventListener('input', function () { renderMetrics(); });
});

$playBtn.addEventListener('click', togglePlay);
$stepBtn.addEventListener('click', step);
$resetBtn.addEventListener('click', reset);
$rerouteBtn.addEventListener('click', function () {
  state.seed = MoEEngine.generateSeed();
  rerouteAndRender();
});

// Scrubber: seek through the wave when dragged.
$scrubber.addEventListener('input', function () {
  state.playing = false;
  updatePlayBtn();
  state.started = true;
  var pct = parseFloat($scrubber.value);
  state.waveElapsed = (pct / 100) * TOTAL_DUR;
  renderFrame();
});

function togglePlay() {
  if (!state.started) { state.started = true; state.waveElapsed = 0; }
  state.playing = !state.playing;
  updatePlayBtn();
  if (state.playing) {
    lastFrame = performance.now();
    requestAnimationFrame(tick);
  }
}
function updatePlayBtn() {
  $playBtn.innerHTML = state.playing
    ? '<svg class="btn-icon" viewBox="0 0 12 12" fill="currentColor"><rect x="2" y="1.5" width="3" height="9" rx="0.5"/><rect x="7" y="1.5" width="3" height="9" rx="0.5"/></svg> Pause'
    : '<svg class="btn-icon" viewBox="0 0 12 12" fill="currentColor"><path d="M3 1.5 L10 6 L3 10.5 Z"/></svg> Play';
  $playBtn.classList.toggle('primary', state.playing);
}
function step() {
  state.playing = false;
  updatePlayBtn();
  state.started = true;
  // Step to the END of the next phase (informative snapshots).
  var curIdx = currentPhaseIndex();
  var nextIdx = curIdx + 1;
  if (nextIdx >= PHASES.length) { reset(); return; }
  var cum = 0;
  for (var i = 0; i <= nextIdx; i++) cum += PHASES[i].dur;
  state.waveElapsed = cum - 1;
  renderFrame();
}
function reset() {
  state.playing = false;
  state.started = false;
  state.waveElapsed = 0;
  updatePlayBtn();
  $scrubber.value = 0;
  renderFrame();
}

/* ───────────────────────────────────────────────────────────────
   Routing + full render
   ─────────────────────────────────────────────────────────────── */
function computeRoute() {
  state.route = MoEEngine.route({
    numGpus: state.numGpus,
    expertsPerGpu: state.expertsPerGpu,
    numTokens: state.numTokens,
    topk: state.topk,
    skew: state.skew,
    seed: state.seed
  });
}
function rerouteAndRender() {
  computeRoute();
  renderVizStructure();
  renderMetrics();
  renderAffinity();
  renderFrame();
}

/* ───────────────────────────────────────────────────────────────
   Phase helpers
   ─────────────────────────────────────────────────────────────── */
function currentPhaseIndex() {
  if (!state.started) return -1;
  var e = state.waveElapsed;
  if (e >= TOTAL_DUR) return PHASES.length - 1;
  var acc = 0;
  for (var i = 0; i < PHASES.length; i++) {
    acc += PHASES[i].dur;
    if (e < acc) return i;
  }
  return PHASES.length - 1;
}
function phaseProgress() {
  var idx = currentPhaseIndex();
  if (idx < 0) return 0;
  var e = state.waveElapsed;
  var acc = 0;
  for (var i = 0; i < idx; i++) acc += PHASES[i].dur;
  var dur = PHASES[idx].dur;
  return dur > 0 ? Math.max(0, Math.min(1, (e - acc) / dur)) : 0;
}
function easeOut(t) { return 1 - (1 - t) * (1 - t); }
function easeIn(t) { return t * t; }

/* ───────────────────────────────────────────────────────────────
   SVG namespace helpers
   ─────────────────────────────────────────────────────────────── */
var SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl(name, attrs, text) {
  var el = document.createElementNS(SVG_NS, name);
  if (attrs) for (var k in attrs) el.setAttribute(k, attrs[k]);
  if (text != null) el.textContent = text;
  return el;
}
function clearNode(n) { while (n.firstChild) n.removeChild(n.firstChild); }

/* ───────────────────────────────────────────────────────────────
   Layout geometry for the visualization
   ─────────────────────────────────────────────────────────────── */
var GEO = {
  marginX: 50,
  row1Y: 60,
  row2Y: 210,
  row3Y: 360,
  nodeW: 80,
  nodeH: 90,
  bucketW: 24,
  bucketGap: 4,
  svgBaseH: 440
};

function layoutGpus() {
  var G = state.numGpus;
  var w = Math.max(520, G * 120);
  var usable = w - GEO.marginX * 2;
  var step = G > 1 ? usable / (G - 1) : 0;
  var xs = [];
  for (var i = 0; i < G; i++) xs.push(GEO.marginX + i * step);
  return { w: w, h: GEO.svgBaseH, xs: xs, step: step };
}

/* ───────────────────────────────────────────────────────────────
   Build static SVG structure (nodes, buckets, connection paths).
   Stores references for per-frame updates.
   ─────────────────────────────────────────────────────────────── */
var pathRefs = {};
var combinePathRefs = {};
var bucketRefs = [];
var sendTankRefs = [];
var outputTankRefs = [];

function renderVizStructure() {
  if (!state.route) return;
  refreshPalette();
  var R = state.route;
  var G = R.numGpus;
  var E = R.expertsPerGpu;
  var L = layoutGpus();
  pathRefs = {};
  combinePathRefs = {};
  bucketRefs = [];
  sendTankRefs = [];
  outputTankRefs = [];

  $vizSvg.setAttribute('viewBox', '0 0 ' + L.w + ' ' + L.h);
  $vizSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  clearNode($vizSvg);

  var defs = svgEl('defs');
  defs.appendChild(svgEl('filter', { id: 'buckGlow', x: '-50%', y: '-50%', width: '200%', height: '200%' }));
  var fe = svgEl('feGaussianBlur', { stdDeviation: '2.5', result: 'b' });
  defs.querySelector('#buckGlow').appendChild(fe);
  var merge = svgEl('feMerge');
  merge.appendChild(svgEl('feMergeNode', { in: 'b' }));
  merge.appendChild(svgEl('feMergeNode', { in: 'SourceGraphic' }));
  defs.querySelector('#buckGlow').appendChild(merge);
  $vizSvg.appendChild(defs);

  var connLayer = svgEl('g', { id: 'connLayer' });
  var combineConnLayer = svgEl('g', { id: 'combineConnLayer' });
  var particleLayer = svgEl('g', { id: 'particleLayer' });
  var wmLayer = svgEl('g', { id: 'wmLayer' });
  var gpuLayer = svgEl('g', { id: 'gpuLayer' });
  var bucketLayer = svgEl('g', { id: 'bucketLayer' });
  var overlayLayer = svgEl('g', { id: 'overlayLayer' });
  $vizSvg.appendChild(connLayer);
  $vizSvg.appendChild(combineConnLayer);
  $vizSvg.appendChild(particleLayer);
  $vizSvg.appendChild(wmLayer);
  $vizSvg.appendChild(gpuLayer);
  $vizSvg.appendChild(bucketLayer);
  $vizSvg.appendChild(overlayLayer);

  var y1Bot = GEO.row1Y + GEO.nodeH / 2;
  var y2Top = GEO.row2Y - GEO.nodeH / 2;
  var y2Bot = GEO.row2Y + GEO.nodeH / 2;
  var y3Top = GEO.row3Y - GEO.nodeH / 2;

  for (var a = 0; a < G; a++) {
    for (var b = 0; b < G; b++) {
      var xa = L.xs[a], xb = L.xs[b];
      var d = 'M' + xa + ',' + y1Bot + ' L' + xb + ',' + y2Top;
      var path = svgEl('path', {
        d: d, fill: 'none',
        stroke: palette.text3, 'stroke-width': '0.6',
        'stroke-linecap': 'round', opacity: '0.08',
        'data-i': a, 'data-j': b
      });
      var title = svgEl('title');
      title.textContent = 'GPU ' + a + ' \u2192 GPU ' + b + ': ' + R.dispatchMatrix[a][b] + ' tokens';
      path.appendChild(title);
      connLayer.appendChild(path);
      var midX = (xa + xb) / 2;
      var midY = (y1Bot + y2Top) / 2;
      var lbl = svgEl('text', {
        x: midX, y: midY, 'text-anchor': 'middle',
        class: 'conn-label', 'data-i': a, 'data-j': b
      });
      lbl.textContent = '0';
      lbl.setAttribute('opacity', '0');
      connLayer.appendChild(lbl);
      try { pathRefs[a + '_' + b] = { el: path, len: path.getTotalLength(), label: lbl, title: title }; } catch (e) {}
    }
  }

  for (var ca = 0; ca < G; ca++) {
    for (var cb = 0; cb < G; cb++) {
      var cxa = L.xs[ca], cxb = L.xs[cb];
      var cd = 'M' + cxa + ',' + y2Bot + ' L' + cxb + ',' + y3Top;
      var cpath = svgEl('path', {
        d: cd, fill: 'none',
        stroke: palette.text3, 'stroke-width': '0.6',
        'stroke-linecap': 'round', opacity: '0.08',
        'data-i': ca, 'data-j': cb, class: 'combine-conn'
      });
      var ctitle = svgEl('title');
      ctitle.textContent = 'GPU ' + ca + ' \u2192 GPU ' + cb + ': ' + R.combineMatrix[ca][cb] + ' tokens (combine)';
      cpath.appendChild(ctitle);
      combineConnLayer.appendChild(cpath);
      var cmidX = (cxa + cxb) / 2;
      var cmidY = (y2Bot + y3Top) / 2;
      var clbl = svgEl('text', {
        x: cmidX, y: cmidY, 'text-anchor': 'middle',
        class: 'conn-label', 'data-i': ca, 'data-j': cb
      });
      clbl.textContent = '0';
      clbl.setAttribute('opacity', '0');
      combineConnLayer.appendChild(clbl);
      try { combinePathRefs[ca + '_' + cb] = { el: cpath, len: cpath.getTotalLength(), label: clbl, title: ctitle }; } catch (e) {}
    }
  }

  var rowLabels = ['SEND', 'RECV', 'OUTPUT'];
  for (var rl = 0; rl < 3; rl++) {
    var ry = [GEO.row1Y, GEO.row2Y, GEO.row3Y][rl];
    var rlbl = svgEl('text', { x: 14, y: ry + 3, class: 'row-label' });
    rlbl.textContent = rowLabels[rl];
    wmLayer.appendChild(rlbl);
  }

  var halfW = GEO.nodeW / 2;
  var halfH = GEO.nodeH / 2;

  for (var gi = 0; gi < G; gi++) {
    var x = L.xs[gi];
    var nx = x - halfW;
    var ny = GEO.row1Y - halfH;
    var bg = svgEl('rect', {
      x: nx, y: ny, width: GEO.nodeW, height: GEO.nodeH, rx: 6,
      fill: palette.surface2, stroke: palette.border, 'stroke-width': '1', opacity: '0.6'
    });
    var bgt = svgEl('title');
    bgt.textContent = 'GPU ' + gi + ' send volume: 0 tokens';
    bg.appendChild(bgt);
    gpuLayer.appendChild(bg);
    var wfill = svgEl('rect', {
      x: nx, y: GEO.row1Y + halfH, width: GEO.nodeW, height: 0, rx: 6,
      fill: palette.green, opacity: '0.7'
    });
    gpuLayer.appendChild(wfill);
    var lbl = svgEl('text', { x: x, y: ny - 5, 'text-anchor': 'middle', class: 'gpu-label' });
    lbl.textContent = 'GPU ' + gi;
    gpuLayer.appendChild(lbl);
    var cnt = svgEl('text', { x: x, y: GEO.row1Y + halfH + 12, 'text-anchor': 'middle', class: 'gpu-counter', id: 'send_' + gi });
    cnt.textContent = '\u2191 0';
    gpuLayer.appendChild(cnt);
    sendTankRefs[gi] = { fill: wfill, bg: bg, title: bgt, x: nx, y: ny, w: GEO.nodeW, h: GEO.nodeH };
  }

  var totalBW = E * GEO.bucketW + (E - 1) * GEO.bucketGap;
  var   bucketAreaH = GEO.nodeH - 34;
  for (var g2 = 0; g2 < G; g2++) {
    var x2 = L.xs[g2];
    var nx2 = x2 - halfW;
    var ny2 = GEO.row2Y - halfH;
    var gbg = svgEl('rect', {
      x: nx2, y: ny2, width: GEO.nodeW, height: GEO.nodeH, rx: 6,
      fill: palette.surface2, stroke: palette.border, 'stroke-width': '1', opacity: '0.5'
    });
    var gbt = svgEl('title');
    gbt.textContent = 'GPU ' + g2 + ' (expert compute)';
    gbg.appendChild(gbt);
    gpuLayer.appendChild(gbg);
    var gl2 = svgEl('text', { x: x2, y: ny2 - 5, 'text-anchor': 'middle', class: 'gpu-label' });
    gl2.textContent = 'GPU ' + g2;
    gpuLayer.appendChild(gl2);

    var startX = x2 - totalBW / 2;
    var bucketTop = ny2 + 7;
    for (var e = 0; e < E; e++) {
      var bx = startX + e * (GEO.bucketW + GEO.bucketGap);
      var by = bucketTop;
      var ebg = svgEl('rect', {
        x: bx, y: by, width: GEO.bucketW, height: bucketAreaH, rx: 2,
        fill: palette.surface2, opacity: '0.6'
      });
      var bt = svgEl('title');
      bt.textContent = 'GPU ' + g2 + ' Expert ' + e + ': 0 tokens';
      ebg.appendChild(bt);
      bucketLayer.appendChild(ebg);
      var efill = svgEl('rect', {
        x: bx, y: by + bucketAreaH, width: GEO.bucketW, height: 0, rx: 2,
        fill: palette.green, 'data-gpu': g2, 'data-exp': e
      });
      bucketLayer.appendChild(efill);
      var count = svgEl('text', { x: bx + GEO.bucketW / 2, y: by + bucketAreaH + 11, 'text-anchor': 'middle', class: 'bucket-count' });
      count.textContent = '0';
      bucketLayer.appendChild(count);
      var elbl = svgEl('text', { x: bx + GEO.bucketW / 2, y: by + bucketAreaH + 22, 'text-anchor': 'middle', class: 'bucket-label' });
      elbl.textContent = 'e' + (g2 * E + e);
      bucketLayer.appendChild(elbl);
      bucketRefs.push({ gpu: g2, exp: e, fill: efill, count: count, bg: ebg, title: bt, x: bx, y: by, load: R.expertLoads[g2][e], areaH: bucketAreaH });
    }
  }

  for (var g3 = 0; g3 < G; g3++) {
    var x3 = L.xs[g3];
    var nx3 = x3 - halfW;
    var ny3 = GEO.row3Y - halfH;
    var obg = svgEl('rect', {
      x: nx3, y: ny3, width: GEO.nodeW, height: GEO.nodeH, rx: 6,
      fill: palette.surface2, stroke: palette.border, 'stroke-width': '1', opacity: '0.6'
    });
    var obt = svgEl('title');
    obt.textContent = 'GPU ' + g3 + ' output volume: 0 tokens';
    obg.appendChild(obt);
    gpuLayer.appendChild(obg);
    var ofill = svgEl('rect', {
      x: nx3, y: GEO.row3Y + halfH, width: GEO.nodeW, height: 0, rx: 6,
      fill: palette.green, opacity: '0.7'
    });
    gpuLayer.appendChild(ofill);
    var ol3 = svgEl('text', { x: x3, y: ny3 - 5, 'text-anchor': 'middle', class: 'gpu-label' });
    ol3.textContent = 'GPU ' + g3;
    gpuLayer.appendChild(ol3);
    var ocnt = svgEl('text', { x: x3, y: GEO.row3Y + halfH + 12, 'text-anchor': 'middle', class: 'gpu-counter', id: 'output_' + g3 });
    ocnt.textContent = '\u2193 0';
    gpuLayer.appendChild(ocnt);
    outputTankRefs[g3] = { fill: ofill, bg: obg, title: obt, x: nx3, y: ny3, w: GEO.nodeW, h: GEO.nodeH };
  }

  for (var oi = 0; oi < G; oi++) {
    (function(gpu, x, ny) {
      var ov = svgEl('rect', { x: x - halfW, y: ny, width: GEO.nodeW, height: GEO.nodeH, rx: 6, fill: 'transparent', 'pointer-events': 'all', style: 'cursor:help' });
      ov.addEventListener('mouseenter', function() { showTooltip(sendTooltipHtml(gpu), ov); });
      ov.addEventListener('mouseleave', hideTooltip);
      overlayLayer.appendChild(ov);
    })(oi, L.xs[oi], GEO.row1Y - halfH);
  }

  for (var bi = 0; bi < bucketRefs.length; bi++) {
    (function(ref) {
      var ov = svgEl('rect', { x: ref.x - 2, y: ref.y - 2, width: GEO.bucketW + 4, height: ref.areaH, rx: 3, fill: 'transparent', 'pointer-events': 'all', style: 'cursor:help' });
      ov.addEventListener('mouseenter', function() { showTooltip(expertTooltipHtml(ref.gpu, ref.exp), ov); });
      ov.addEventListener('mouseleave', hideTooltip);
      overlayLayer.appendChild(ov);
    })(bucketRefs[bi]);
  }

  for (var oo = 0; oo < G; oo++) {
    (function(gpu, x, ny3) {
      var ov = svgEl('rect', { x: x - halfW, y: ny3, width: GEO.nodeW, height: GEO.nodeH, rx: 6, fill: 'transparent', 'pointer-events': 'all', style: 'cursor:help' });
      ov.addEventListener('mouseenter', function() { showTooltip(outputTooltipHtml(gpu), ov); });
      ov.addEventListener('mouseleave', hideTooltip);
      overlayLayer.appendChild(ov);
    })(oo, L.xs[oo], GEO.row3Y - halfH);
  }

  renderLegend();
}

function renderLegend() {
  var items = [
    { c: palette.green, t: 'low load' },
    { c: palette.orange, t: 'medium' },
    { c: palette.red, t: 'overloaded' },
    { c: palette.accent, t: 'particle' }
  ];
  var html = '';
  items.forEach(function (it) {
    html += '<span class="leg-item"><span class="leg-dot" style="background:' + it.c + '"></span>' + it.t + '</span>';
  });
  $vizLegend.innerHTML = html;
}

/* ───────────────────────────────────────────────────────────────
   Particle plan (built from route, used in send & combine)
   Particles are a pure function of phase progress -> seek-friendly.
   ─────────────────────────────────────────────────────────────── */
function buildParticlePlan(useCombine) {
  var R = state.route;
  if (!R) return [];
  var G = R.numGpus;
  var plan = [];
  var matrix = useCombine ? R.combineMatrix : R.dispatchMatrix;
  var refs = useCombine ? combinePathRefs : pathRefs;
  var totalMoved = R.totalTokensMoved;
  var scale = Math.max(1, totalMoved / 120);
  var travelSpan = 0.75;
  for (var i = 0; i < G; i++) {
    for (var j = 0; j < G; j++) {
      if (i === j) continue;
      var count = matrix[i][j];
      if (count <= 0) continue;
      var key = i + '_' + j;
      var ref = refs[key];
      if (!ref) continue;
      var n = Math.max(1, Math.round(count / scale));
      var color = srcColor(i);
      for (var k = 0; k < n; k++) {
        var spawnFrac = 0.02 + (n > 1 ? (k / (n - 1)) * 0.2 : 0.05);
        plan.push({ ref: ref, spawnFrac: spawnFrac, travelSpan: travelSpan, color: color });
      }
    }
  }
  return plan;
}

function drawDispatchParticles(layer, pp) {
  var R = state.route;
  if (!R) return;
  var G = R.numGpus;
  var L = layoutGpus();
  var y1Top = GEO.row1Y - GEO.nodeH / 2;
  var startY = 6;
  for (var i = 0; i < G; i++) {
    var x = L.xs[i];
    var nPerGpu = 5;
    for (var k = 0; k < nPerGpu; k++) {
      var offset = (k / nPerGpu) * 0.5;
      var prog = ((pp * 2 + offset) % 1);
      var py = startY + prog * (y1Top - startY);
      var px = x + Math.sin(k * 2.5 + i * 1.7) * 5;
      var op = 0.3 + prog * 0.5;
      var c = svgEl('circle', { cx: px.toFixed(1), cy: py.toFixed(1), r: '2.5', fill: srcColor(i), opacity: op.toFixed(2) });
      layer.appendChild(c);
    }
  }
}

/* ───────────────────────────────────────────────────────────────
   Per-frame render: particles, bucket fills, connections, counters
   ─────────────────────────────────────────────────────────────── */
var lastPlan = null;
var lastPlanReverse = false;

function renderFrame() {
  if (!state.route) return;
  var R = state.route;
  var started = state.started;
  var idx = currentPhaseIndex();
  var pp = phaseProgress();
  var phaseKey = idx >= 0 ? PHASES[idx].key : 'idle';

  // phase bar
  renderPhaseBar(idx, pp);
  // scrubber
  var pct = started ? Math.min(100, (state.waveElapsed / TOTAL_DUR) * 100) : 0;
  $scrubber.value = pct;
  $scrubFrom.textContent = started ? (state.waveElapsed / 1000).toFixed(1) + 's' : '0.0s';
  $scrubTo.textContent = (TOTAL_DUR / 1000).toFixed(1) + 's';

  var descBar = document.getElementById('phaseDescBar');
  if (descBar) {
    var descs = {
      'dispatch': '\u3010Dispatch \u8def\u7531\u89c4\u5212\u3011\u8f93\u5165 token \u7ecf\u8fc7 router gate \u8bc4\u5206\uff0c\u4e3a\u6bcf\u4e2a token \u9009\u51fa top-k \u4e2a\u4e13\u5bb6\uff0c\u5e76\u751f\u6210\u5206\u53d1\u8ba1\u5212\u3002\u4e0a\u65b9\u6c34\u4f4d\u4e0a\u5347 = \u5404 GPU \u5373\u5c06\u53d1\u9001\u7684 token \u6570\u3002',
      'send': '\u3010Send \u5206\u53d1\u3011All-to-All \u9636\u6bb5\uff1atoken \u4ece SEND \u6d41\u5411 RECV\uff08\u4e13\u5bb6\u5c42\uff09\u3002\u6bcf\u4e2a GPU \u5c06\u5176 token \u5206\u53d1\u5230\u76ee\u6807\u4e13\u5bb6\u6240\u5728\u7684 GPU\uff0c\u7c92\u5b50\u6d41\u5c55\u793a\u4f20\u8f93\u65b9\u5411\u4e0e\u6d41\u91cf\u3002',
      'compute': '\u3010Compute \u4e13\u5bb6\u8ba1\u7b97\u3011\u5404\u4e13\u5bb6\u72ec\u7acb\u6267\u884c FFN \u8ba1\u7b97\u3002\u8f7d\u8377\u4f4e\u7684\u4e13\u5bb6\u63d0\u524d\u5b8c\u6210\uff0c\u70ed\u70b9\u4e13\u5bb6\u662f\u74f6\u9888\u3002\u6876\u586b\u8272\u8d8a\u7ea2 = \u8f7d\u8377\u8d8a\u91cd\u3002',
      'combine': '\u3010Combine \u6c47\u5408\u3011All-to-All \u9636\u6bb5\uff1a\u4e13\u5bb6\u8ba1\u7b97\u7ed3\u679c\u4ece RECV \u6d41\u5411 OUTPUT\uff0c\u53d1\u56de\u7ed9\u539f\u59cb GPU\u3002\u4e0b\u65b9\u6c34\u4f4d\u4e0a\u5347 = \u5404 GPU \u6536\u56de\u7684\u7ed3\u679c\u6570\u3002'
    };
    descBar.textContent = started ? (descs[phaseKey] || '') : '\u70b9\u51fb Play \u5f00\u59cb\u64ad\u653e\uff0c\u89c2\u5bdf MoE All-to-All \u901a\u4fe1\u7684\u56db\u4e2a\u9636\u6bb5\u3002';
    descBar.className = 'phase-desc-bar' + (started ? ' active phase-' + phaseKey : '');
  }

  // connection visibility / dispatch labels
  var showConn = started && (phaseKey === 'dispatch' || phaseKey === 'send' || phaseKey === 'compute' || phaseKey === 'combine');
  var labelOpacity = 0;
  if (started && phaseKey === 'dispatch') labelOpacity = easeOut(pp);
  else if (started) labelOpacity = 0.85;
  updateConnections(showConn, labelOpacity, phaseKey);
  updateCombineConnections(started && (phaseKey === 'compute' || phaseKey === 'combine'), phaseKey);

  var particleLayer = document.getElementById('particleLayer');
  clearNode(particleLayer);
  if (started && phaseKey === 'dispatch') {
    drawDispatchParticles(particleLayer, pp);
    lastPlan = null;
  } else if (started && (phaseKey === 'send' || phaseKey === 'combine')) {
    var useCombine = phaseKey === 'combine';
    if (!lastPlan || lastPlanReverse !== useCombine) {
      lastPlan = buildParticlePlan(useCombine);
      lastPlanReverse = useCombine;
    }
    drawParticles(particleLayer, lastPlan, pp);
  } else {
    lastPlan = null;
  }

  updateBuckets(started, phaseKey, pp, R);
  updateTanks(started, phaseKey, pp, R);

  $affinityChart.classList.toggle('pulsing', started && phaseKey === 'dispatch');
}

function updateConnections(show, labelOpacity, phaseKey) {
  var R = state.route;
  if (!R) return;
  var G = R.numGpus;
  var maxCount = 1, sumCount = 0, nNonZero = 0;
  for (var a = 0; a < G; a++) {
    for (var b = 0; b < G; b++) {
      var c = R.dispatchMatrix[a][b];
      if (c > maxCount) maxCount = c;
      if (c > 0) { sumCount += c; nNonZero++; }
    }
  }
  var avgCount = nNonZero > 0 ? sumCount / nNonZero : 0;
  for (var key in pathRefs) {
    var ref = pathRefs[key];
    var parts = key.split('_');
    var i = parseInt(parts[0], 10), j = parseInt(parts[1], 10);
    var count = R.dispatchMatrix[i][j];
    var ratio = count / maxCount;
    var isHot = avgCount > 0 && count > 2 * avgCount;
    var baseOp = show ? (0.06 + ratio * 0.35) : 0.06;
    if (phaseKey === 'send' || phaseKey === 'combine') baseOp = Math.max(baseOp, 0.2 + ratio * 0.3);
    ref.el.setAttribute('opacity', baseOp.toFixed(2));
    ref.el.setAttribute('stroke', isHot && show ? palette.red : palette.text3);
    var sw = 0.5 + ratio * 1.2;
    if (isHot && show) sw += 0.4;
    ref.el.setAttribute('stroke-width', sw.toFixed(2));
    ref.label.textContent = String(count);
    ref.label.setAttribute('opacity', show ? (labelOpacity * (isHot ? 1 : 0.6)).toFixed(2) : '0');
    ref.label.setAttribute('class', 'conn-label' + (isHot ? ' hot' : ''));
    ref.title.textContent = 'GPU ' + i + ' \u2192 GPU ' + j + ': ' + count + ' tokens';
  }
}

function updateCombineConnections(show, phaseKey) {
  var R = state.route;
  if (!R) return;
  var G = R.numGpus;
  var maxCount = 1, sumCount = 0, nNonZero = 0;
  for (var a = 0; a < G; a++) {
    for (var b = 0; b < G; b++) {
      var c = R.combineMatrix[a][b];
      if (c > maxCount) maxCount = c;
      if (c > 0) { sumCount += c; nNonZero++; }
    }
  }
  var avgCount = nNonZero > 0 ? sumCount / nNonZero : 0;
  for (var key in combinePathRefs) {
    var ref = combinePathRefs[key];
    var parts = key.split('_');
    var i = parseInt(parts[0], 10), j = parseInt(parts[1], 10);
    var count = R.combineMatrix[i][j];
    var ratio = count / maxCount;
    var isHot = avgCount > 0 && count > 2 * avgCount;
    var baseOp = show ? (0.06 + ratio * 0.35) : 0.06;
    if (phaseKey === 'combine') baseOp = Math.max(baseOp, 0.2 + ratio * 0.3);
    ref.el.setAttribute('opacity', baseOp.toFixed(2));
    ref.el.setAttribute('stroke', isHot && show ? palette.red : palette.text3);
    var sw = 0.5 + ratio * 1.2;
    if (isHot && show) sw += 0.4;
    ref.el.setAttribute('stroke-width', sw.toFixed(2));
    ref.label.textContent = String(count);
    ref.label.setAttribute('opacity', show ? (0.85 * (isHot ? 1 : 0.6)).toFixed(2) : '0');
    ref.label.setAttribute('class', 'conn-label' + (isHot ? ' hot' : ''));
    ref.title.textContent = 'GPU ' + i + ' \u2192 GPU ' + j + ': ' + count + ' tokens (combine)';
  }
}

function drawParticles(layer, plan, pp) {
  for (var k = 0; k < plan.length; k++) {
    var p = plan[k];
    var prog = (pp - p.spawnFrac) / p.travelSpan;
    if (prog <= 0 || prog >= 1) continue;
    var frac = prog;
    var len = p.ref.len * frac;
    var pt;
    try { pt = p.ref.el.getPointAtLength(len); } catch (e) { continue; }
    var c = svgEl('circle', { cx: pt.x.toFixed(1), cy: pt.y.toFixed(1), r: '3.2', fill: p.color, opacity: '0.9' });
    layer.appendChild(c);
  }
}

function updateBuckets(started, phaseKey, pp, R) {
  var maxLoad = R.maxExpertLoad || 1;
  var avg = R.avgExpertLoad || 0;
  for (var b = 0; b < bucketRefs.length; b++) {
    var ref = bucketRefs[b];
    var target = ref.load;
    var fillRatio = 0;
    if (!started) fillRatio = 0;
    else if (phaseKey === 'dispatch') fillRatio = 0;
    else if (phaseKey === 'send') fillRatio = easeOut(pp) * (target / maxLoad);
    else if (phaseKey === 'compute') fillRatio = target / maxLoad;
    else if (phaseKey === 'combine') fillRatio = (target / maxLoad) * (1 - easeIn(pp));
    fillRatio = Math.max(0, Math.min(1, fillRatio));
    var fh = fillRatio * ref.areaH;
    ref.fill.setAttribute('y', (ref.y + ref.areaH - fh).toFixed(1));
    ref.fill.setAttribute('height', fh.toFixed(1));
    var overloaded = avg > 0 && target > 1.5 * avg;
    var color = loadColor(target, maxLoad, avg);
    ref.fill.setAttribute('fill', color);

    var computeProgress = 0;
    var computeDone = false;
    if (phaseKey === 'compute' && target > 0 && maxLoad > 0) {
      computeProgress = pp * (maxLoad / target);
      computeDone = computeProgress >= 1;
    }

    if (phaseKey === 'compute' && target > 0 && !computeDone) {
      ref.fill.setAttribute('filter', overloaded ? 'url(#buckGlow)' : '');
      ref.fill.style.animation = 'bucketGlow 1.4s ease-in-out infinite';
      ref.fill.setAttribute('opacity', '1');
    } else if (phaseKey === 'compute' && computeDone) {
      ref.fill.removeAttribute('filter');
      ref.fill.style.animation = '';
      ref.fill.setAttribute('opacity', '0.35');
    } else {
      ref.fill.removeAttribute('filter');
      ref.fill.style.animation = '';
      ref.fill.setAttribute('opacity', '1');
    }

    if (started) {
      ref.count.textContent = String(target);
      ref.count.setAttribute('fill', overloaded ? palette.red : palette.text2);
    } else {
      ref.count.textContent = '0';
      ref.count.setAttribute('fill', palette.text3);
    }

    var mult = avg > 0 ? (target / avg).toFixed(1) : '0';
    ref.title.textContent = 'GPU ' + ref.gpu + ' Expert ' + ref.exp + ': ' + target + ' tokens (' + mult + '\u00d7 avg)' +
      (phaseKey === 'compute' && computeDone ? ' \u2713 done' : '');
  }
}

function setTankLevel(tankRef, ratio, color) {
  if (!tankRef) return;
  var h = ratio * tankRef.h;
  tankRef.fill.setAttribute('y', (tankRef.y + tankRef.h - h).toFixed(1));
  tankRef.fill.setAttribute('height', h.toFixed(1));
  tankRef.fill.setAttribute('fill', color);
}

function tankColor(load, avg) {
  if (load <= 0) return palette.surface2;
  if (avg <= 0) return palette.accent;
  var ratio = load / avg;
  if (ratio > 1.5) return palette.red;
  var green = hexToRgb(palette.green);
  var accent = hexToRgb(palette.accent);
  var orange = hexToRgb(palette.orange);
  var c;
  if (ratio < 1.0) c = mixRgb(green, accent, ratio);
  else c = mixRgb(accent, orange, Math.min(1, (ratio - 1.0) / 0.5));
  return rgbStr(c);
}

function updateTanks(started, phaseKey, pp, R) {
  var G = R.numGpus;
  var sendTotals = [];
  var outputTotals = [];
  var maxSend = 0, maxOutput = 0;
  var totalSend = 0, totalOutput = 0;
  for (var g = 0; g < G; g++) {
    var s = 0, o = 0;
    for (var d = 0; d < G; d++) {
      s += R.dispatchMatrix[g][d];
      o += R.combineMatrix[d][g];
    }
    sendTotals[g] = s;
    outputTotals[g] = o;
    totalSend += s;
    totalOutput += o;
    if (s > maxSend) maxSend = s;
    if (o > maxOutput) maxOutput = o;
  }
  var avgSend = totalSend / G;
  var avgOutput = totalOutput / G;
  for (var i = 0; i < G; i++) {
    var sendRatio = 0;
    if (!started) sendRatio = 0;
    else if (phaseKey === 'dispatch') sendRatio = easeOut(pp) * (sendTotals[i] / maxSend);
    else if (phaseKey === 'send') sendRatio = (sendTotals[i] / maxSend) * (1 - easeOut(pp));
    else sendRatio = 0;
    sendRatio = Math.max(0, Math.min(1, sendRatio));
    setTankLevel(sendTankRefs[i], sendRatio, srcColor(i));
    if (sendTankRefs[i] && sendTankRefs[i].title) sendTankRefs[i].title.textContent = 'GPU ' + i + ' send volume: ' + sendTotals[i] + ' tokens';

    var outputRatio = 0;
    if (!started) outputRatio = 0;
    else if (phaseKey === 'combine') outputRatio = easeOut(pp) * (outputTotals[i] / maxOutput);
    outputRatio = Math.max(0, Math.min(1, outputRatio));
    setTankLevel(outputTankRefs[i], outputRatio, tankColor(outputTotals[i], avgOutput));
    if (outputTankRefs[i] && outputTankRefs[i].title) outputTankRefs[i].title.textContent = 'GPU ' + i + ' output volume: ' + outputTotals[i] + ' tokens';

    var sendTxt = document.getElementById('send_' + i);
    var outputTxt = document.getElementById('output_' + i);
    if (sendTxt) sendTxt.textContent = (started ? (R.tokenCounts[i] + ' \u2192 ' + sendTotals[i]) : '');
    if (outputTxt) outputTxt.textContent = '\u2193 ' + (started && phaseKey === 'combine' ? Math.round(outputTotals[i] * easeOut(pp)) : 0);
  }
}

function renderPhaseBar(idx, pp) {
  var chips = $phaseBar.querySelectorAll('.phase-chip');
  chips.forEach(function (chip, i) {
    chip.classList.remove('active', 'done');
    if (!state.started) return;
    if (i < idx) chip.classList.add('done');
    else if (i === idx) { chip.classList.add('active'); chip.style.setProperty('--phase-progress', (pp * 100).toFixed(0) + '%'); }
    else chip.style.setProperty('--phase-progress', '0%');
    if (i === idx && idx === PHASES.length - 1 && pp >= 0.999) chip.classList.add('done');
  });
}

/* ───────────────────────────────────────────────────────────────
   Metrics panel
   ─────────────────────────────────────────────────────────────── */
function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(Math.round(n));
}
function fmtDec(n, d) { return n.toFixed(d); }

function renderMetrics() {
  if (!state.route) return;
  var R = state.route;
  var tokBytes = state.tokenSizeKB * 1024;
  var commSec = MoEEngine.estimateCommTime(R.totalTokensRoundTrip, tokBytes, state.bandwidthGBs);
  var compSec = MoEEngine.estimateComputeTime(R.maxExpertLoad, state.expertSpeedK * 1000);
  var ratio = compSec > 0 ? commSec / compSec : 0;

  // imbalance hero
  var imb = R.imbalanceRatio;
  $imbalanceValue.textContent = imb.toFixed(2) + '\u00d7';
  $imbalanceValue.classList.remove('lvl-mid', 'lvl-high');
  if (imb > 2) { $imbalanceValue.classList.add('lvl-high'); $imbalanceHint.textContent = 'severe imbalance'; }
  else if (imb > 1.5) { $imbalanceValue.classList.add('lvl-mid'); $imbalanceHint.textContent = 'moderate imbalance'; }
  else { $imbalanceHint.textContent = 'well balanced'; }

  var rows = '';
  rows += mrow('Total tokens', fmt(R.numTokens));
  rows += mrow('Tokens moved (one-way)', fmt(R.totalTokensMoved), 'accent');
  rows += mrow('Tokens moved (round-trip)', fmt(R.totalTokensRoundTrip), 'accent');
  rows += mrow('Max expert load', fmt(R.maxExpertLoad), R.maxExpertLoad > 1.5 * R.avgExpertLoad ? 'bad' : '');
  rows += mrow('Min expert load', fmt(R.minExpertLoad));
  rows += mrow('Avg expert load', fmtDec(R.avgExpertLoad, 1));
  rows += mrow('Max GPU receive', fmt(R.maxGpuRecvLoad), 'warn');
  rows += mrow('Max GPU send', fmt(R.maxGpuSendLoad), 'warn');
  rows += mrow('Est. comm time', fmtDec(commSec * 1000, 2) + ' ms');
  rows += mrow('Est. compute time', fmtDec(compSec * 1000, 2) + ' ms');
  rows += mrow('Comm / Compute', fmtDec(ratio, 2), ratio > 1 ? 'bad' : 'good');
  $metricList.innerHTML = rows;

  $hotCard.innerHTML =
    '<span class="hot-tag">Hot Expert</span>GPU ' + R.hotExpert.gpu + ' &middot; expert ' + R.hotExpert.expert +
    ' &rarr; <span class="hot-val">' + fmt(R.hotExpert.load) + ' tok</span><br>' +
    '<span class="hot-tag">Hot GPU</span>GPU ' + R.hotGpu.gpu + ' &middot; receiving <span class="hot-val">' + fmt(R.hotGpu.recvLoad) + ' tok</span>' +
    ' &middot; sending ' + fmt(R.hotGpu.sendLoad) + ' tok';
}
function mrow(label, val, cls) {
  return '<div class="metric-row"><span class="m-label">' + label +
    '</span><span class="m-val ' + (cls || '') + '">' + val + '</span></div>';
}

/* ───────────────────────────────────────────────────────────────
   Affinity chart (popularityWeights) — explains WHY imbalance exists
   ─────────────────────────────────────────────────────────────── */
function renderAffinity() {
  if (!state.route) return;
  var R = state.route;
  var w = R.popularityWeights;
  var max = 0;
  for (var i = 0; i < w.length; i++) if (w[i] > max) max = w[i];
  var html = '';
  for (var k = 0; k < w.length; k++) {
    var h = max > 0 ? (w[k] / max) * 100 : 0;
    var hot = max > 0 && w[k] > max * 0.75;
    html += '<div class="affinity-bar' + (hot ? ' hot' : '') + '" style="height:' + h.toFixed(1) + '%" title="expert ' + k + ': ' + h.toFixed(1) + '% rel."></div>';
  }
  $affinityChart.innerHTML = html;
  $affinityAxis.innerHTML = '<span>e0</span><span>experts ' + w.length + ' total &middot; GPU' + R.numGpus + '\u00d7' + R.expertsPerGpu + '</span><span>e' + (w.length - 1) + '</span>';
}

/* ───────────────────────────────────────────────────────────────
   Animation loop
   ─────────────────────────────────────────────────────────────── */
var lastFrame = 0;
function tick(now) {
  if (!state.playing) return;
  var dt = now - lastFrame;
  lastFrame = now;
  state.waveElapsed += dt * state.speed;
  if (state.waveElapsed >= TOTAL_DUR) {
    // loop continuously
    state.waveElapsed = state.waveElapsed % TOTAL_DUR;
  }
  renderFrame();
  requestAnimationFrame(tick);
}

/* ───────────────────────────────────────────────────────────────
   Init
   ─────────────────────────────────────────────────────────────── */
syncScenarioActive();
updatePlayBtn();
computeRoute();
renderVizStructure();
renderMetrics();
renderAffinity();
renderFrame();

// re-layout on resize (SVG scales, but counters/positions are viewBox-based so just re-render)
var resizeT = null;
window.addEventListener('resize', function () {
  if (resizeT) clearTimeout(resizeT);
  resizeT = setTimeout(function () { renderVizStructure(); renderFrame(); }, 200);
});
