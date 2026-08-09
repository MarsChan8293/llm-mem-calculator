if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

var $modelPicker = document.getElementById('modelPicker');
var $selectedTags = document.getElementById('selectedTags');
var $wtPrecSeg = document.getElementById('wtPrecSeg');
var $kvPrecSeg = document.getElementById('kvPrecSeg');
var $idxPrecSeg = document.getElementById('idxPrecSeg');
var $idxPrecField = document.getElementById('idxPrecField');
var $unifiedParallel = document.getElementById('unifiedParallel');
var $ctxInput = document.getElementById('ctxInput');
var $ctxPresets = document.getElementById('ctxPresets');
var $batchInput = document.getElementById('batchInput');
var $batchPresets = document.getElementById('batchPresets');
var $draftToggle = document.getElementById('draftToggle');
var $draftField = document.getElementById('draftField');
var $linearToggle = document.getElementById('linearToggle');
var $linearField = document.getElementById('linearField');
var $tpInput = document.getElementById('tpInput');
var $ppInput = document.getElementById('ppInput');
var $epInput = document.getElementById('epInput');
var $epItem = document.getElementById('epItem');
var $cpInput = document.getElementById('cpInput');
var $dpInput = document.getElementById('dpInput');
var $idxTpInput = document.getElementById('idxTpInput');
var $idxTpField = document.getElementById('idxTpField');
var $idxTpHint = document.getElementById('idxTpHint');
var $modeSeg = document.getElementById('modeSeg');
var $disaggControls = document.getElementById('disaggControls');
var $prefillTpInput = document.getElementById('prefillTpInput');
var $prefillEpInput = document.getElementById('prefillEpInput');
var $prefillPpInput = document.getElementById('prefillPpInput');
var $prefillCpInput = document.getElementById('prefillCpInput');
var $prefillDpInput = document.getElementById('prefillDpInput');
var $prefillEpItem = document.getElementById('prefillEpItem');
var $decodeTpInput = document.getElementById('decodeTpInput');
var $decodeEpInput = document.getElementById('decodeEpInput');
var $decodePpInput = document.getElementById('decodePpInput');
var $decodeCpInput = document.getElementById('decodeCpInput');
var $decodeDpInput = document.getElementById('decodeDpInput');
var $decodeEpItem = document.getElementById('decodeEpItem');
var $absorptionToggle = document.getElementById('absorptionToggle');
var $absorptionField = document.getElementById('absorptionField');
var $gpuSelect = document.getElementById('gpuSelect');
var $totalPerGpu = document.getElementById('totalPerGpu');
var $totalUnit = document.getElementById('totalUnit');
var $ibarSection = document.getElementById('ibarSection');
var $gpuFitSection = document.getElementById('gpuFitSection');
var $metricsCompact = document.getElementById('metricsCompact');
var $formulaSection = document.getElementById('formulaSection');
var $formulaTitle = document.getElementById('formulaTitle');
var $formulaBody = document.getElementById('formulaBody');
var $topologySection = document.getElementById('topologySection');
var $stageSection = document.getElementById('stageSection');
var $breakdownGrid = document.getElementById('breakdownGrid');
var $breakdownToggle = document.getElementById('breakdownToggle');
var $noteSection = document.getElementById('noteSection');
var $sourceLink = document.getElementById('sourceLink');
var $themeToggle = document.getElementById('themeToggle');

var selectedModelId = 'deepseek-v4-pro';
var wtPrecValue = 'fp8_int8';
var kvPrecValue = 'bf16';
var idxPrecValue = 'bf16';
var servingMode = 'unified';

function initTheme() {
  var stored = localStorage.getItem('kv-theme');
  if (stored === 'dark' || stored === 'light') {
    document.documentElement.setAttribute('data-theme', stored);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
}

function toggleTheme() {
  var current = document.documentElement.getAttribute('data-theme');
  var next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('kv-theme', next);
}

initTheme();
$themeToggle.addEventListener('click', toggleTheme);

function getPrecBytes(val) {
  var opts = MODEL_DATA.weight_precision_options;
  if (!opts) return 2;
  return opts.find(function (p) { return p.id === val; }).bytes_per_element;
}

function getModel() {
  if (!selectedModelId) return null;
  return MODEL_DATA.models.find(function (m) { return m.id === selectedModelId; });
}

var families = [];
var familyMap = {};
MODEL_DATA.models.forEach(function (m) {
  if (!familyMap[m.family]) {
    familyMap[m.family] = [];
    families.push(m.family);
  }
  familyMap[m.family].push(m);
});

var collapsedFamilies = {};

function buildPicker(filter) {
  var searchInput = $modelPicker.querySelector('.picker-search');
  var hadFocus = searchInput && document.activeElement === searchInput;
  var cursorPos = searchInput ? searchInput.selectionStart : 0;

  while ($modelPicker.lastChild) $modelPicker.removeChild($modelPicker.lastChild);

  if (!searchInput) {
    searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.className = 'picker-search';
    searchInput.placeholder = 'Search models...';
    searchInput.addEventListener('input', function () { buildPicker(searchInput.value); });
  }
  if (filter !== undefined) searchInput.value = filter;
  $modelPicker.appendChild(searchInput);

  var query = (searchInput.value || '').toLowerCase().trim();

  families.forEach(function (fam) {
    var models = familyMap[fam].filter(function (m) {
      if (!query) return true;
      return m.label.toLowerCase().indexOf(query) !== -1 ||
             m.family.toLowerCase().indexOf(query) !== -1 ||
             m.id.toLowerCase().indexOf(query) !== -1;
    });
    if (models.length === 0) return;

    var header = document.createElement('div');
    header.className = 'picker-family';
    header.textContent = (collapsedFamilies[fam] ? '\u25B6 ' : '\u25BC ') + fam;
    header.addEventListener('click', function () {
      collapsedFamilies[fam] = !collapsedFamilies[fam];
      buildPicker(searchInput.value);
    });
    $modelPicker.appendChild(header);

    if (!collapsedFamilies[fam]) {
      models.forEach(function (m) {
        var item = document.createElement('div');
        item.className = 'picker-item';
        if (selectedModelId === m.id) item.classList.add('selected');
        var nameSpan = document.createElement('span');
        nameSpan.textContent = m.label;
        var addBtn = document.createElement('span');
        addBtn.className = 'picker-add';
        addBtn.textContent = selectedModelId === m.id ? '\u25CF' : '\u25CB';
        item.appendChild(nameSpan);
        item.appendChild(addBtn);
        item.addEventListener('click', function () {
          if (selectedModelId !== m.id) {
            selectedModelId = m.id;
            applyModelDefaults();
            renderTag();
            buildPicker(searchInput.value);
            updateConditionalFields();
            calculate();
          }
        });
        $modelPicker.appendChild(item);
      });
    }
  });

  if (hadFocus) {
    searchInput.focus();
    searchInput.setSelectionRange(cursorPos, cursorPos);
  }
}

function renderTag() {
  $selectedTags.innerHTML = '';
  if (!selectedModelId) return;
  var m = getModel();
  if (!m) return;
  var accentColor = getCSSVar('--accent') || '#6366f1';
  var accentLight = getCSSVar('--accent-light') || 'rgba(99,102,241,0.12)';
  var tag = document.createElement('span');
  tag.className = 'tag';
  tag.style.background = accentLight;
  tag.style.color = accentColor;
  var dot = document.createElement('span');
  dot.className = 'tag-dot';
  dot.style.background = accentColor;
  var label = document.createElement('span');
  label.textContent = m.label;
  var removeBtn = document.createElement('span');
  removeBtn.className = 'tag-remove';
  removeBtn.textContent = '\u00d7';
  removeBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    selectedModelId = null;
    renderTag();
    var si = $modelPicker.querySelector('.picker-search');
    buildPicker(si ? si.value : '');
    calculate();
  });
  tag.appendChild(dot);
  tag.appendChild(label);
  tag.appendChild(removeBtn);
  $selectedTags.appendChild(tag);
}

function getCSSVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function applyModelDefaults() {
  var model = getModel();
  if (!model) return;
  var defaults = getDeployDefaults(model);
  $tpInput.value = defaults.tp;
  $ppInput.value = defaults.pp;
  $epInput.value = defaults.ep;
  $dpInput.value = defaults.dp;
  $idxTpInput.value = defaults.idxTp;
  $prefillTpInput.value = defaults.tp;
  $prefillEpInput.value = defaults.ep;
  $decodeTpInput.value = Math.max(1, Math.floor(defaults.tp / 4));
  $decodeEpInput.value = defaults.ep;

  var gpuOpt = GPU_OPTIONS.find(function (g) { return g.id === defaults.gpu; });
  if (gpuOpt) $gpuSelect.value = defaults.gpu;
}

function updateConditionalFields() {
  var model = getModel();
  if (!model) return;
  var wf = model.weight_fields || {};
  var hasMoE = (wf.n_routed_experts || 0) > 0;
  var hasIndexer = modelHasIndexer(model);
  var hasAbsorption = modelSupportsAbsorption(model);
  var hasDraft = !!(model.fields.mtp_transformer_layers);
  var hasLinear = ['qwen_linear_full_hybrid', 'kda_gated_mla'].includes(model.formula);

  $epItem.style.display = hasMoE ? '' : 'none';
  $idxPrecField.style.display = hasIndexer ? '' : 'none';
  $idxTpField.style.display = hasIndexer ? '' : 'none';
  if (hasIndexer) {
    $idxTpHint.textContent = '(Indexer typically runs TP=1)';
  }
  $draftField.style.display = hasDraft ? '' : 'none';
  $linearField.style.display = hasLinear ? '' : 'none';
  if (hasLinear && model.formula === 'kda_gated_mla') {
    $linearToggle.checked = true;
  }
  $absorptionField.style.display = (hasAbsorption && servingMode === 'disaggregated') ? '' : 'none';
  $prefillEpItem.style.display = hasMoE ? '' : 'none';
  $decodeEpItem.style.display = hasMoE ? '' : 'none';
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

initSegControl($wtPrecSeg, function (val) { wtPrecValue = val; calculate(); });
initSegControl($kvPrecSeg, function (val) { kvPrecValue = val; calculate(); });
initSegControl($idxPrecSeg, function (val) { idxPrecValue = val; calculate(); });
initSegControl($modeSeg, function (val) {
  servingMode = val;
  $unifiedParallel.style.display = val === 'unified' ? '' : 'none';
  $disaggControls.style.display = val === 'disaggregated' ? '' : 'none';
  updateConditionalFields();
  calculate();
});

$ctxPresets.querySelectorAll('.preset-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    $ctxPresets.querySelectorAll('.preset-btn').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    $ctxInput.value = btn.getAttribute('data-value');
    calculate();
  });
});

$ctxInput.addEventListener('input', function () {
  $ctxPresets.querySelectorAll('.preset-btn').forEach(function (b) { b.classList.remove('active'); });
  calculate();
});

$batchPresets.querySelectorAll('.preset-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    $batchPresets.querySelectorAll('.preset-btn').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    $batchInput.value = btn.getAttribute('data-value');
    calculate();
  });
});

$batchInput.addEventListener('input', function () {
  $batchPresets.querySelectorAll('.preset-btn').forEach(function (b) { b.classList.remove('active'); });
  calculate();
});

[$tpInput, $ppInput, $epInput, $cpInput, $dpInput, $idxTpInput,
 $prefillTpInput, $prefillPpInput, $prefillEpInput, $prefillCpInput, $prefillDpInput,
 $decodeTpInput, $decodePpInput, $decodeEpInput, $decodeCpInput, $decodeDpInput].forEach(function (el) {
  el.addEventListener('input', function () { calculate(); });
});

$draftToggle.addEventListener('change', function () { calculate(); });
$linearToggle.addEventListener('change', function () { calculate(); });
$absorptionToggle.addEventListener('change', function () { calculate(); });

GPU_OPTIONS.forEach(function (g) {
  var opt = document.createElement('option');
  opt.value = g.id;
  opt.textContent = g.label;
  $gpuSelect.appendChild(opt);
});
$gpuSelect.addEventListener('change', function () { calculate(); });

function formatTotal(bytes) {
  return (bytes / 1e9).toFixed(5);
}

function getUnitLabel() {
  return 'GB';
}

function formatMetric(bytes) {
  return (bytes / 1e9).toFixed(5) + ' GB';
}

function fmtNum(n) { return n.toLocaleString('en-US'); }

function fmtWNum(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function fmtWBytes(bytes) { return formatMetric(bytes); }

function tipIcon(tooltip) {
  return '<span class="tip-icon" data-tooltip="' + tooltip.replace(/"/g, '&quot;') + '">?</span>';
}

function formatSymbol(text) {
  return text.replace(/^([A-Za-z]+)_(.+)$/, '$1<sub>$2</sub>');
}

function getBarHex(type) { return DEPLOY_BAR_HEX_MAP[type] || '#4263eb'; }
function getBarLabel(type) { return DEPLOY_LEGEND_MAP[type] || type; }

function renderIbar(segments, maxBytes) {
  if (!segments || segments.length === 0) return '';
  var html = '<div class="ibar">';
  segments.forEach(function (seg) {
    var w = maxBytes > 0 ? Math.max(1, (seg.bytes / maxBytes) * 200) : 1;
    html += '<div class="seg seg-deploy-' + seg.type + '" style="width:' + w + 'px;background:' + getBarHex(seg.type) + '" data-tooltip="' + getBarLabel(seg.type) + ': ' + formatMetric(seg.bytes) + '"></div>';
  });
  html += '</div>';
  html += '<div class="ibar-legend">';
  var shown = {};
  segments.forEach(function (seg) {
    if (!shown[seg.type]) {
      shown[seg.type] = true;
      html += '<span class="ibar-legend-item"><span class="ibar-legend-dot" style="background:' + getBarHex(seg.type) + '"></span>' + getBarLabel(seg.type) + '</span>';
    }
  });
  html += '</div>';
  return html;
}

function renderGpuFit(fit) {
  var pct = (fit.usage * 100).toFixed(1);
  var status, statusIcon, barColor;
  if (fit.usage < 0.7) {
    status = 'ok';
    statusIcon = '\u2705';
    barColor = '#40c057';
  } else if (fit.usage < 0.9) {
    status = 'warn';
    statusIcon = '\u26a0\ufe0f';
    barColor = '#f59e0b';
  } else {
    status = 'oom';
    statusIcon = '\u274c';
    barColor = '#e03131';
  }
  var usedGB = (fit.usedVram / 1e9).toFixed(2);
  var usableGB = (fit.usableVram / 1e9).toFixed(2);
  var totalGB = (fit.vram / 1e9).toFixed(2);
  var overheadGB = (fit.fixedOverhead / 1e9).toFixed(2);
  var html = '<div class="gpu-fit-card ' + status + '">';
  html += '<div class="gpu-fit-header">' + fit.label + '</div>';
  html += '<div class="gpu-fit-bar-wrap"><div class="gpu-fit-bar" style="width:' + Math.min(100, pct) + '%;background:' + barColor + '"></div></div>';
  html += '<div class="gpu-fit-stats">' + usedGB + ' / ' + usableGB + ' GB usable ' + statusIcon + ' ' + pct + '%</div>';
  html += '<div class="gpu-fit-note">' + totalGB + ' GB × ' + (fit.utilizationLimit * 100).toFixed(0) + '% − ' + overheadGB + ' GB CUDA Graph reserve</div>';
  html += '</div>';
  return html;
}

function renderStage(stage, maxBytes) {
  var html = '<div class="stage-card">';
  html += '<div class="stage-header">Stage ' + stage.stageIndex + ' <span class="stage-range">(' + stage.layerRange + ': ';
  var parts = [];
  if (stage.denseLayers > 0) parts.push(stage.denseLayers + ' dense');
  if (stage.moeLayers > 0) parts.push(stage.moeLayers + ' MoE');
  html += parts.join(', ') + ')</span></div>';
  html += renderIbar(stage.ibar, maxBytes);
  html += '<div class="stage-total">' + formatMetric(stage.totalPerGPU) + ' per GPU</div>';
  html += '</div>';
  return html;
}

function renderFormulaRows(formulas) {
  if (!formulas || formulas.length === 0) return '';
  var globalMaxBarBytes = 0;
  formulas.forEach(function (f) {
    if (f.bar) f.bar.forEach(function (seg) { if (seg.bytes > globalMaxBarBytes) globalMaxBarBytes = seg.bytes; });
  });

  return formulas.map(function (f) {
    var expr = f.expr;
    var vals = Object.assign({}, f.values);
    var keys = Object.keys(vals).sort(function (a, b) { return b.length - a.length; });
    if (keys.length > 0) {
      var re = new RegExp('\\b(' + keys.map(function (k) { return k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }).join('|') + ')\\b', 'g');
      expr = expr.replace(re, function (match) {
        var val = vals[match];
        var tooltipText = typeof val === 'number' ? match + ' = ' + fmtNum(val) : val;
        var cls = 'pill pill-param';
        return '<span class="' + cls + '" data-tooltip="' + tooltipText.replace(/"/g, '&quot;') + '">' + formatSymbol(match) + '</span>';
      });
    }
    expr = expr.replace(/\u230a/g, '<span class="floor">\u230a</span>');
    expr = expr.replace(/\u230b/g, '<span class="floor">\u230b</span>');

    var nameTooltip = '';
    if (f.resultValue !== undefined) nameTooltip = formatMetric(f.resultValue);
    if (f.tip) nameTooltip = nameTooltip ? nameTooltip + '\n' + f.tip : f.tip;
    var namePill = '<span class="pill pill-result" data-tooltip="' + nameTooltip.replace(/"/g, '&quot;') + '">' + formatSymbol(f.name) + '</span>';

    var ibarHtml = '';
    if (f.bar && f.bar.length > 0) {
      ibarHtml = '<div class="ibar">';
      f.bar.forEach(function (seg) {
        var w = globalMaxBarBytes > 0 ? Math.max(1, (seg.bytes / globalMaxBarBytes) * 120) : 1;
        ibarHtml += '<div class="seg seg-deploy-' + seg.type + '" style="width:' + w + 'px;background:' + getBarHex(seg.type) + '" data-tooltip="' + getBarLabel(seg.type) + ': ' + formatMetric(seg.bytes) + '"></div>';
      });
      ibarHtml += '</div>';
      ibarHtml += '<div class="ibar-val">' + (f.ibarVal || '') + '</div>';
    }

    return '<div class="formula-row">' +
      '<div class="formula-lhs">' + namePill + '<span class="formula-eq">=</span></div>' +
      '<div class="formula-rhs">' + expr + '</div>' +
      ibarHtml +
    '</div>';
  }).join('');
}

var TOPO_PALETTE = {
  attn: '#0d9488',
  kv: '#6366f1',
  shared: '#ea580c',
  epBase: [175, 70, 72]
};

function epShade(base, g) {
  var h = base[0] + g * 3, s = base[1] - g * 3, l = base[2] - g * 5;
  return 'hsl(' + h + ',' + s + '%,' + l + '%)';
}

function topoGb(bytes) {
  return (bytes / 1e9).toFixed(2) + ' GB';
}

function renderTopology(model, result, opts) {
  var wf = model.weight_fields || {};
  var nRouted = wf.n_routed_experts || 0;
  var nShared = wf.n_shared_experts || 0;
  var wb = result.weightBreakdown;
  var kb = result.kvBreakdown;
  var tp = opts.tp || 1;
  var ep = opts.ep || 1;
  var dp = opts.dp || 1;
  var pp = opts.pp || 1;
  var cp = opts.cp || 1;
  var isMoE = nRouted > 0;
  var hasDense = wb.denseFfnPerGPU > 0;
  var numAttn = Math.min(dp, 3);
  var expertsPerGPU = nRouted > 0 ? Math.ceil(nRouted / ep) : 0;
  var p = TOPO_PALETTE;
  var isDisagg = opts._disaggLabel;

  var html = '';
  if (isDisagg) html += '<div class="topo-disagg-label">' + isDisagg + '</div>';

  html += '<div class="topo">';
  html += '<div class="attn-stack">';

  for (var a = 0; a < numAttn; a++) {
    html += '<div class="block">';
    html += '<div class="block-title"><div class="block-title-dot" style="background:' + p.attn + '"></div>Attention' + (dp > 1 ? ' ' + (a + 1) : '') + '</div>';

    html += '<div class="row">';
    html += '<div class="row-label">W<sub>qkv</sub> <span class="row-hint">TP column split (by heads)</span> <span class="row-gb">' + topoGb(wb.attnPerGPU) + '</span></div>';
    html += '<div class="col-bar" style="height:32px">';
    for (var g = 0; g < tp; g++) {
      html += '<div class="col-shard" style="background:' + p.attn + '" data-tooltip="GPU ' + g + ': QKV shard ' + (g + 1) + '/' + tp + '">' + g + '</div>';
    }
    html += '</div>';
    html += '</div>';

    html += '<div class="row">';
    html += '<div class="row-label">W<sub>o</sub> <span class="row-hint">TP row split (by heads)</span></div>';
    html += '<div class="stripe-bar" style="height:32px;gap:1px">';
    for (var g = 0; g < tp; g++) {
      html += '<div class="stripe-seg" style="background:' + p.attn + '" data-tooltip="GPU ' + g + ': Wo shard ' + (g + 1) + '/' + tp + '"></div>';
    }
    html += '</div>';
    html += '</div>';

    html += '<div class="row">';
    html += '<div class="row-label">KV Cache <span class="row-hint">TP split (by heads)</span> <span class="row-gb">' + topoGb(kb.kvPerGPU) + '</span></div>';
    html += '<div class="kv-bar" style="height:16px">';
    for (var g = 0; g < tp; g++) {
      html += '<div class="kv-shard" style="background:' + p.kv + '" data-tooltip="GPU ' + g + ': KV shard ' + (g + 1) + '/' + tp + '">' + g + '</div>';
    }
    html += '</div>';
    html += '</div>';

    html += '</div>';
  }

  html += '</div>';

  html += '<div class="gather" id="gatherZone' + (isDisagg || '') + '"></div>';

  html += '<div class="block block-moe">';
  var moeTitle = isMoE ? 'MLP / MoE' : 'MLP';
  html += '<div class="block-title"><div class="block-title-dot" style="background:' + p.shared + '"></div>' + moeTitle + '</div>';

  if (hasDense) {
    html += '<div class="row">';
    html += '<div class="row-label">Dense FFN <span class="row-hint">TP col + row</span> <span class="row-gb">' + topoGb(wb.denseFfnPerGPU) + '</span></div>';
    html += '<div class="col-bar" style="height:14px">';
    for (var g = 0; g < tp; g++) {
      html += '<div class="col-shard" style="background:' + p.shared + '" data-tooltip="GPU ' + g + ': Dense Gate/Up shard"></div>';
    }
    html += '</div>';
    html += '<div class="stripe-bar" style="height:14px;margin-top:2px;gap:1px">';
    for (var g = 0; g < tp; g++) {
      html += '<div class="stripe-seg" style="background:' + p.shared + '" data-tooltip="GPU ' + g + ': Dense Down shard"></div>';
    }
    html += '</div>';
    html += '</div>';
  }

  if (isMoE) {
    if (nShared > 0) {
      html += '<div class="row">';
      html += '<div class="row-label">Shared Expert <span class="row-hint">TP col + row</span> <span class="row-gb">' + topoGb(wb.sharedExpertPerGPU) + '</span></div>';
      html += '<div class="col-bar" style="height:14px">';
      for (var g = 0; g < tp; g++) {
        html += '<div class="col-shard" style="background:' + p.shared + '" data-tooltip="GPU ' + g + ': Shared Gate/Up shard"></div>';
      }
      html += '</div>';
      html += '<div class="stripe-bar" style="height:14px;margin-top:2px;gap:1px">';
      for (var g = 0; g < tp; g++) {
        html += '<div class="stripe-seg" style="background:' + p.shared + '" data-tooltip="GPU ' + g + ': Shared Down shard"></div>';
      }
      html += '</div>';
      html += '</div>';
    }

    html += '<div class="row">';
    html += '<div class="row-label">Routed Experts <span class="row-hint">EP=' + ep + ' groups, TP col+row inside each</span> <span class="row-gb" style="color:var(--ep)">' + topoGb(wb.routedExpertPerGPU) + '</span></div>';
    html += '<div class="ep-groups">';
    for (var g = 0; g < ep; g++) {
      var eStart = g * expertsPerGPU + 1;
      var eEnd = (g + 1) * expertsPerGPU;
      var shade = epShade(p.epBase, g);
      html += '<div class="ep-group" style="background:var(--surface);border-color:' + shade + '">';
      html += '<div class="ep-header" style="color:' + shade + '">E' + eStart + '&ndash;' + eEnd + '</div>';
      html += '<div class="ep-gpu">GPU ' + g + '</div>';
      html += '<div class="ep-bar-label">G/U</div>';
      html += '<div class="mini-col">';
      for (var t = 0; t < tp; t++) {
        html += '<div class="mini-col-s" style="background:' + shade + '" data-tooltip="GPU ' + g + ': Gate/Up TP' + t + '"></div>';
      }
      html += '</div>';
      html += '<div class="ep-bar-label">Down</div>';
      html += '<div class="mini-row">';
      for (var t = 0; t < tp; t++) {
        html += '<div class="mini-row-s" style="background:' + shade + '" data-tooltip="GPU ' + g + ': Down TP' + t + '"></div>';
      }
      html += '</div>';
      html += '</div>';
    }
    html += '</div>';
    html += '</div>';
  }

  html += '</div>';
  html += '</div>';

  if (pp > 1) {
    html += '<div class="topo-note-card"><span class="topo-note-badge topo-note-pp">PP=' + pp + '</span>PP cut happens between layers, not within a layer</div>';
  }
  if (cp > 1) {
    html += '<div class="topo-note-card"><span class="topo-note-badge topo-note-cp">CP=' + cp + '</span>CP&gt;1 divides KV along sequence-length dimension across CP ranks.</div>';
  }

  html += '<div class="topo-note">Showing bottleneck stage per-GPU weight partitioning.' + (isMoE ? ' EP=' + ep + ' gives each GPU ' + expertsPerGPU + '/' + nRouted + ' experts.' : '') + ' TP=' + tp + ' splits attention' + (isMoE ? ', shared expert' : '') + ', and embedding across ' + tp + ' ranks.</div>';

  return html;
}

function renderDisaggTopology(model, result, prefillOpts, decodeOpts) {
  var pre = result.prefill;
  var dec = result.decode;

  var html = '<div class="topo-section">';
  html += '<div class="topo-title">Per-Layer Weight Partitioning</div>';
  html += '<div class="topo-disagg-grid">';

  html += '<div class="topo-disagg-panel">';
  html += renderTopology(model, pre, Object.assign({}, prefillOpts, { _disaggLabel: 'Prefill' }));
  html += '</div>';

  html += '<div class="topo-disagg-panel">';
  html += renderTopology(model, dec, Object.assign({}, decodeOpts, { _disaggLabel: 'Decode' }));
  html += '</div>';

  html += '</div>';
  html += '</div>';

  return html;
}

function drawGather() {
  var zones = document.querySelectorAll('.gather');
  zones.forEach(function (zone) {
    var stack = zone.previousElementSibling;
    var moe = zone.nextElementSibling;
    if (!stack || !moe) return;
    var zr = zone.getBoundingClientRect();
    var mr = moe.getBoundingClientRect();
    var moeMidY = mr.top + mr.height / 2 - zr.top;
    var attnBlocks = stack.querySelectorAll('.block');
    var w = zr.width;
    var h = zr.height;
    var svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" fill="none" style="position:absolute;inset:0;width:100%;height:100%">';
    var sw = 'stroke="var(--text3)" stroke-width="1.2"';
    var bendX = Math.round(w * 0.4);
    attnBlocks.forEach(function (blk) {
      var br = blk.getBoundingClientRect();
      var fromY = br.top + br.height / 2 - zr.top;
      var toY = moeMidY;
      if (Math.abs(fromY - toY) < 2) {
        svg += '<line x1="0" y1="' + fromY.toFixed(1) + '" x2="' + (w - 6).toFixed(1) + '" y2="' + toY.toFixed(1) + '" ' + sw + '/>';
      } else {
        svg += '<path d="M0,' + fromY.toFixed(1) + ' L' + bendX + ',' + fromY.toFixed(1) + ' L' + bendX + ',' + toY.toFixed(1) + ' L' + (w - 6).toFixed(1) + ',' + toY.toFixed(1) + '" ' + sw + '/>';
      }
    });
    svg += '<polygon points="' + (w - 6) + ',' + (moeMidY - 3).toFixed(1) + ' ' + w + ',' + moeMidY.toFixed(1) + ' ' + (w - 6) + ',' + (moeMidY + 3).toFixed(1) + '" fill="var(--text3)"/>';
    svg += '</svg>';
    zone.innerHTML = svg;
  });
}

function calculate() {
  var model = getModel();
  if (!model) {
    $totalPerGpu.textContent = '\u2014';
    $totalUnit.textContent = getUnitLabel();
    $ibarSection.innerHTML = '';
    $gpuFitSection.innerHTML = '';
    $metricsCompact.innerHTML = '';
    $formulaSection.classList.add('hidden');
    $topologySection.innerHTML = '';
    $stageSection.innerHTML = '';
    $breakdownGrid.innerHTML = '';
    $noteSection.textContent = '';
    $sourceLink.href = '#';
    $sourceLink.textContent = '';
    return;
  }

  var opts = {
    wtPrecB: getPrecBytes(wtPrecValue),
    kvPrecB: getPrecBytes(kvPrecValue),
    idxB: getPrecBytes(idxPrecValue),
    tokens: parseInt($ctxInput.value) || 1024,
    batch: parseInt($batchInput.value) || 1,
    tp: parseInt($tpInput.value) || 1,
    pp: parseInt($ppInput.value) || 1,
    ep: parseInt($epInput.value) || 1,
    dp: parseInt($dpInput.value) || 1,
    idxTp: parseInt($idxTpInput.value) || parseInt($tpInput.value) || 1,
    includeDraft: $draftToggle.checked,
    includeLinear: $linearToggle.checked,
    gpuId: $gpuSelect.value,
    mode: servingMode,
  };

  if (servingMode === 'disaggregated') {
    opts.prefill = {
      tp: parseInt($prefillTpInput.value) || 1,
      pp: parseInt($prefillPpInput.value) || 1,
      ep: parseInt($prefillEpInput.value) || 1,
      cp: parseInt($prefillCpInput.value) || 1,
      dp: parseInt($prefillDpInput.value) || 1,
    };
    opts.decode = {
      tp: parseInt($decodeTpInput.value) || 1,
      pp: parseInt($decodePpInput.value) || 1,
      ep: parseInt($decodeEpInput.value) || 1,
      cp: parseInt($decodeCpInput.value) || 1,
      dp: parseInt($decodeDpInput.value) || 1,
      absorption: $absorptionToggle.checked,
    };
  }

  var result = calcDeploy(model, opts);

  if (result.mode === 'disaggregated') {
    renderDisaggregated(result, model, opts);
  } else {
    renderUnified(result, model, opts);
  }
}

function renderUnified(result, model, opts) {
  $totalPerGpu.textContent = formatTotal(result.totalPerGPU);
  $totalUnit.textContent = getUnitLabel();

  var maxIbar = result.totalPerGPU;
  $ibarSection.innerHTML = renderIbar(result.ibarSegments, maxIbar);

  $gpuFitSection.innerHTML = renderGpuFit(result.gpuFit);

  var wb = result.weightBreakdown;
  var kb = result.kvBreakdown;
  var metricsHtml = '';
  metricsHtml += '<span class="metric-item">Weights <span class="metric-val">' + formatMetric(result.weightPerGPU) + '</span></span>';
  metricsHtml += '<span class="metric-sep">\u00b7</span>';
  metricsHtml += '<span class="metric-item">KV <span class="metric-val">' + formatMetric(result.kvPerGPU) + '</span></span>';
  metricsHtml += '<span class="metric-sep">\u00b7</span>';
  metricsHtml += '<span class="metric-item">GPUs <span class="metric-val">' + result.totalGPUs + '</span></span>';
  $metricsCompact.innerHTML = metricsHtml;

  if (result.formulas && result.formulas.length > 0) {
    $formulaSection.classList.remove('hidden');
    $formulaTitle.textContent = result.formulaTitle;
    $formulaBody.innerHTML = renderFormulaRows(result.formulas);
  } else {
    $formulaSection.classList.add('hidden');
  }

  $topologySection.innerHTML = '<div class="topo-section"><div class="topo-title">Per-Layer Weight Partitioning</div>' + renderTopology(model, result, opts) + '</div>';
  requestAnimationFrame(drawGather);

  if (result.stages && result.stages.length > 1) {
    var maxStageBytes = 0;
    result.stages.forEach(function (s) { if (s.totalPerGPU > maxStageBytes) maxStageBytes = s.totalPerGPU; });
    var stageHtml = '<div class="stage-section-title">Layer Distribution</div>';
    result.stages.forEach(function (s) {
      var isBottleneck = s.stageIndex === result.bottleneckStageIndex;
      stageHtml += renderStage(s, maxStageBytes);
      if (isBottleneck && result.stages.length > 1) {
        stageHtml += '<div class="stage-bottleneck">\u26a0\ufe0f Stage ' + s.stageIndex + ' is the bottleneck</div>';
      }
    });
    $stageSection.innerHTML = stageHtml;
  } else {
    $stageSection.innerHTML = '';
  }

  $breakdownGrid.innerHTML = buildBreakdownRows(result);

  $noteSection.textContent = 'Per-GPU estimates use \u00f7TP for attention/dense/shared-expert/embed, \u00f7EP for routed experts. GPU Fit reserves ' + ((1 - VLLM_GPU_MEMORY_UTILIZATION) * 100).toFixed(0) + '% VRAM for vLLM headroom and ' + VLLM_CUDA_GRAPH_OVERHEAD_GB + ' GB per GPU for CUDA Graph capture. Indexer TP may differ from model TP. Activations, framework overhead, and communication buffers remain excluded.';
  $sourceLink.href = model.source_url;
  $sourceLink.textContent = 'Source: ' + model.source_url;
}

function renderDisaggregated(result, model, opts) {
  var pre = result.prefill;
  var dec = result.decode;

  $totalPerGpu.textContent = '\u2014';
  $totalUnit.textContent = '';
  $ibarSection.innerHTML = '';

  var html = '<div class="disagg-container">';
    html += '<div class="disagg-panel"><div class="disagg-label">Prefill (TP=' + opts.prefill.tp + ', PP=' + opts.prefill.pp + ', EP=' + opts.prefill.ep + ')</div>';
  html += '<div class="disagg-total">' + formatMetric(pre.totalPerGPU) + '</div>';
  html += renderIbar(pre.ibarSegments, pre.totalPerGPU);
  html += renderGpuFit(pre.gpuFit);
  html += '<div class="disagg-metrics">Weights: ' + formatMetric(pre.weightPerGPU) + ' \u00b7 KV: ' + formatMetric(pre.kvPerGPU) + ' \u00b7 GPUs: ' + pre.totalGPUs + '</div>';
  html += '</div>';
    html += '<div class="disagg-panel"><div class="disagg-label">Decode (TP=' + opts.decode.tp + ', PP=' + opts.decode.pp + ', EP=' + opts.decode.ep + ')</div>';
  html += '<div class="disagg-total">' + formatMetric(dec.totalPerGPU) + '</div>';
  html += renderIbar(dec.ibarSegments, dec.totalPerGPU);
  html += renderGpuFit(dec.gpuFit);
  html += '<div class="disagg-metrics">Weights: ' + formatMetric(dec.weightPerGPU) + ' \u00b7 KV: ' + formatMetric(dec.kvPerGPU) + ' \u00b7 GPUs: ' + dec.totalGPUs + '</div>';
  html += '</div>';
  html += '</div>';

  $gpuFitSection.innerHTML = html;
  $metricsCompact.innerHTML = '';
  $formulaSection.classList.add('hidden');

  $topologySection.innerHTML = renderDisaggTopology(model, result, opts.prefill, opts.decode);
  requestAnimationFrame(drawGather);

  $stageSection.innerHTML = '';

  $breakdownGrid.innerHTML = buildBreakdownRows(pre) + '<div class="breakdown-sep"></div>' + buildBreakdownRows(dec);

  $noteSection.textContent = 'Disaggregated deployment: prefill and decode use separate GPU clusters with different parallelism. GPU Fit reserves ' + ((1 - VLLM_GPU_MEMORY_UTILIZATION) * 100).toFixed(0) + '% VRAM for vLLM headroom and ' + VLLM_CUDA_GRAPH_OVERHEAD_GB + ' GB per GPU for CUDA Graph capture. Activations, framework overhead, and communication buffers remain excluded.';
  $sourceLink.href = model.source_url;
  $sourceLink.textContent = 'Source: ' + model.source_url;
}

function buildBreakdownRows(result) {
  var wb = result.weightBreakdown;
  var kb = result.kvBreakdown;
  var rows = [];
  rows.push({ label: 'Attention weights', value: formatMetric(wb.attnPerGPU) });
  if (wb.denseFfnPerGPU > 0) rows.push({ label: 'Dense FFN', value: formatMetric(wb.denseFfnPerGPU) });
  if (wb.sharedExpertPerGPU > 0) rows.push({ label: 'Shared expert', value: formatMetric(wb.sharedExpertPerGPU) });
  if (wb.routedExpertPerGPU > 0) rows.push({ label: 'Routed experts', value: formatMetric(wb.routedExpertPerGPU) });
  rows.push({ label: 'Embedding', value: formatMetric(wb.embedPerGPU) });
  rows.push({ label: 'Weight total', value: formatMetric(result.weightPerGPU) });
  if (kb.kvPerGPU > 0) rows.push({ label: 'KV Cache', value: formatMetric(kb.kvPerGPU) });
  if (kb.idxPerGPU > 0) rows.push({ label: 'Indexer KV', value: formatMetric(kb.idxPerGPU) });
  rows.push({ label: 'Total per GPU', value: formatMetric(result.totalPerGPU) });
  return rows.map(function (r) {
    return '<div class="breakdown-row"><span class="label">' + r.label + '</span><span class="val">' + r.value + '</span></div>';
  }).join('');
}

$breakdownToggle.addEventListener('click', function () {
  var isOpen = $breakdownToggle.classList.toggle('open');
  $breakdownGrid.classList.toggle('collapsed', !isOpen);
});

buildPicker('');
applyModelDefaults();
renderTag();
updateConditionalFields();
calculate();
