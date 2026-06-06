var $modelPicker = document.getElementById('modelPicker');
var $selectedTags = document.getElementById('selectedTags');
var $tokens = document.getElementById('tokens');
var $tokensSlider = document.getElementById('tokensSlider');
var $seq = document.getElementById('seq');
var $precSeg = document.getElementById('precSeg');
var $idxPrecSeg = document.getElementById('idxPrecSeg');
var $idxPrecField = document.getElementById('idxPrecField');
var $draftField = document.getElementById('draftField');
var $linearField = document.getElementById('linearField');
var $draftToggle = document.getElementById('draftToggle');
var $linearToggle = document.getElementById('linearToggle');
var $draftHint = document.getElementById('draftHint');
var $totalValue = document.getElementById('totalValue');
var $totalUnit = document.getElementById('totalUnit');
var $metricsCompact = document.getElementById('metricsCompact');
var $formulaSection = document.getElementById('formulaSection');
var $formulaTitle = document.getElementById('formulaTitle');
var $formulaBody = document.getElementById('formulaBody');
var $breakdownGrid = document.getElementById('breakdownGrid');
var $noteSection = document.getElementById('noteSection');
var $sourceLink = document.getElementById('sourceLink');
var $themeToggle = document.getElementById('themeToggle');

var selectedModelId = 'deepseek-v4-pro';
var currentUnit = 'gib';
var precValue = 'fp8_int8';
var idxPrecValue = 'fp4_int4';

function tipIcon(tooltip) {
  return '<span class="tip-icon" data-tooltip="' + tooltip.replace(/"/g, '&quot;') + '">?</span>';
}

function formatWithCommas(n) {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function parseFormattedNumber(s) {
  return parseInt(s.replace(/,/g, ''), 10) || 0;
}

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
  $modelPicker.innerHTML = '';

  var searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.className = 'picker-search';
  searchInput.placeholder = 'Search models...';
  if (filter) searchInput.value = filter;
  $modelPicker.appendChild(searchInput);

  var query = (filter || '').toLowerCase().trim();

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
          if (selectedModelId === m.id) {
            selectedModelId = null;
          } else {
            selectedModelId = m.id;
          }
          renderTag();
          buildPicker(searchInput.value);
          onModelChange();
        });

        $modelPicker.appendChild(item);
      });
    }
  });

  searchInput.addEventListener('input', function () {
    buildPicker(searchInput.value);
  });
}

function getCSSVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
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
    var searchInput = $modelPicker.querySelector('.picker-search');
    buildPicker(searchInput ? searchInput.value : '');
    onModelChange();
  });

  tag.appendChild(dot);
  tag.appendChild(label);
  tag.appendChild(removeBtn);
  $selectedTags.appendChild(tag);
}

function getModel() {
  if (!selectedModelId) return null;
  return MODEL_DATA.models.find(function (m) { return m.id === selectedModelId; });
}

function getPrecBytes() {
  return MODEL_DATA.precision_options.find(function (p) { return p.id === precValue; }).bytes_per_element;
}

function getIdxPrecBytes() {
  return MODEL_DATA.indexer_precision_options.find(function (p) { return p.id === idxPrecValue; }).bytes_per_element;
}

function onModelChange() {
  var model = getModel();
  if (model) {
    $tokensSlider.max = model.max_position_embeddings;
  }
  calculate();
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

initSegControl($precSeg, function (val) {
  precValue = val;
  calculate();
});

initSegControl($idxPrecSeg, function (val) {
  idxPrecValue = val;
  calculate();
});

$draftToggle.addEventListener('click', function () {
  var current = this.getAttribute('aria-checked') === 'true';
  this.setAttribute('aria-checked', !current);
  calculate();
});

$linearToggle.addEventListener('click', function () {
  var current = this.getAttribute('aria-checked') === 'true';
  this.setAttribute('aria-checked', !current);
  calculate();
});

$tokens.addEventListener('input', function () {
  var raw = parseFormattedNumber(this.value);
  if (isNaN(raw) || raw < 1) raw = 1;
  $tokensSlider.value = raw;
  calculate();
});

$tokens.addEventListener('blur', function () {
  var raw = parseFormattedNumber(this.value);
  if (isNaN(raw) || raw < 1) raw = 1;
  this.value = formatWithCommas(raw);
});

$tokensSlider.addEventListener('input', function () {
  var val = parseInt(this.value, 10);
  $tokens.value = formatWithCommas(val);
  calculate();
});

document.querySelectorAll('.batch-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    document.querySelectorAll('.batch-btn').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    var val = parseInt(btn.getAttribute('data-value'), 10);
    $seq.value = val;
    calculate();
  });
});

$seq.addEventListener('input', function () {
  document.querySelectorAll('.batch-btn').forEach(function (b) {
    b.classList.toggle('active', parseInt(b.getAttribute('data-value'), 10) === parseInt($seq.value, 10));
  });
  calculate();
});

document.querySelectorAll('.unit-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    document.querySelectorAll('.unit-btn').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    currentUnit = btn.getAttribute('data-unit');
    calculate();
  });
});

function formatTotal(bytes) {
  if (currentUnit === 'gib') return (bytes / Math.pow(1024, 3)).toFixed(5);
  if (currentUnit === 'gb') return (bytes / 1e9).toFixed(5);
  if (currentUnit === 'mib') return (bytes / Math.pow(1024, 2)).toFixed(3);
  return bytes;
}

function getUnitLabel() {
  if (currentUnit === 'gib') return 'GiB';
  if (currentUnit === 'gb') return 'GB';
  if (currentUnit === 'mib') return 'MiB';
  return '';
}

function formatMetric(bytes) {
  if (bytes < 1024) return bytes.toFixed(0) + ' B';
  if (bytes < Math.pow(1024, 2)) return (bytes / 1024).toFixed(2) + ' KiB';
  if (bytes < Math.pow(1024, 3)) return (bytes / Math.pow(1024, 2)).toFixed(2) + ' MiB';
  return (bytes / Math.pow(1024, 3)).toFixed(3) + ' GiB';
}

function calculate() {
  var model = getModel();
  if (!model) {
    $totalValue.textContent = '\u2014';
    $totalUnit.textContent = getUnitLabel();
    $metricsCompact.innerHTML = '';
    $formulaSection.classList.add('hidden');
    $breakdownGrid.innerHTML = '';
    $noteSection.textContent = '';
    $sourceLink.href = '#';
    $sourceLink.textContent = '';
    return;
  }

  var tokens = Math.max(1, parseFormattedNumber($tokens.value) || 1);
  var seqs = Math.max(1, parseInt($seq.value) || 1);
  var precB = getPrecBytes();
  var idxB = getIdxPrecBytes();
  var formula = model.formula;
  var f = model.fields;
  var includeDraft = $draftToggle.getAttribute('aria-checked') === 'true';
  var includeLinear = $linearToggle.getAttribute('aria-checked') === 'true';

  var hasIndexer = ['deepseek_v4_hybrid', 'dsa_mla'].includes(formula);
  $idxPrecField.classList.toggle('hidden', !hasIndexer);

  var hasDraft = ['mla', 'dsa_mla', 'deepseek_v4_hybrid', 'standard_gqa'].includes(formula) &&
    (f.num_nextn_predict_layers || f.mtp_transformer_layers);
  $draftField.classList.toggle('hidden', !hasDraft);
  if (hasDraft && formula === 'deepseek_v4_hybrid') {
    $draftHint.textContent = 'Adds model-specific MTP/draft KV layers when enabled by the serving stack. DeepSeek V4 draft layers use ratio=0 sliding-window cache.';
  } else if (hasDraft) {
    $draftHint.textContent = 'Adds model-specific MTP/draft KV layers when enabled by the serving stack.';
  }

  var hasLinear = formula === 'qwen_linear_full_hybrid';
  $linearField.classList.toggle('hidden', !hasLinear);

  var result = calcKvCache(model, tokens, precB, idxB, { includeDraft: includeDraft, includeLinear: includeLinear });

  var totalBytes = seqs * (result.kvBytes + result.idxBytes);

  var lastIdx = result.breakdown.length - 1;
  if (lastIdx >= 0 && result.breakdown[lastIdx].label === 'Total bytes') {
    result.breakdown[lastIdx].value = fmtNum(totalBytes);
  }

  $totalValue.textContent = formatTotal(totalBytes);
  $totalUnit.textContent = getUnitLabel();

  var metricsHtml = '';
  metricsHtml += '<span class="metric-item">KV Cache <span class="metric-val">' + formatMetric(result.kvBytes) + '</span></span>';
  if (result.idxBytes > 0) {
    metricsHtml += '<span class="metric-sep">\u00b7</span>';
    metricsHtml += '<span class="metric-item">Indexer <span class="metric-val">' + formatMetric(result.idxBytes) + '</span></span>';
  }
  metricsHtml += '<span class="metric-sep">\u00b7</span>';
  metricsHtml += '<span class="metric-item">Per Token <span class="metric-val">' + formatMetric(result.perTokenBytes) + '</span></span>';
  $metricsCompact.innerHTML = metricsHtml;

  if (result.formulas.length > 0) {
    $formulaSection.classList.remove('hidden');
    $formulaTitle.textContent = result.formulaTitle;
    $formulaBody.innerHTML = result.formulas.map(function (f) {
      var expr = f.expr;
      var vals = Object.assign({}, f.values);
      if (expr.indexOf('sequences') !== -1) {
        vals.sequences = seqs;
      }
      var inputNames = { tokens: 1, sequences: 1, precision_bytes: 1, indexer_precision_bytes: 1 };
      var keys = Object.keys(vals).sort(function (a, b) { return b.length - a.length; });
      if (keys.length > 0) {
        var re = new RegExp('\\b(' + keys.map(function (k) { return k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }).join('|') + ')\\b', 'g');
        expr = expr.replace(re, function (match) {
          var val = vals[match];
          var tooltipText;
          if (typeof val === 'number') {
            tooltipText = match.indexOf('_bytes') !== -1 ? fmtBytes(val) : fmtNum(val);
          } else {
            tooltipText = val;
          }
          var cls = 'pill';
          if (inputNames[match]) {
            cls += ' pill-input';
          } else if (match.indexOf('_bytes') !== -1) {
            cls += ' pill-result';
          } else {
            cls += ' pill-param';
          }
          return '<span class="' + cls + '" data-tooltip="' + tooltipText.replace(/"/g, '&quot;') + '">' + match + '</span>';
        });
      }
      var nameVal = f.resultValue !== undefined ? f.resultValue : vals[f.name];
      var nameTooltip = '';
      if (nameVal !== undefined) {
        nameTooltip = typeof nameVal === 'number' ? (f.name.indexOf('_bytes') !== -1 ? fmtBytes(nameVal) : fmtNum(nameVal)) : String(nameVal);
      }
      if (f.tip) {
        nameTooltip = nameTooltip ? nameTooltip + '\n' + f.tip : f.tip;
      }
      var nameValText = nameTooltip.split('\n')[0] || '';
      var namePill = '<span class="pill pill-result" data-tooltip="' + nameTooltip.replace(/"/g, '&quot;') + '">' + f.name + '</span>' +
        (nameValText ? '<span class="formula-val">' + nameValText + '</span>' : '');
      return '<div class="formula-row">' +
        '<div class="formula-lhs">' +
          namePill +
          '<span class="formula-eq">=</span>' +
        '</div>' +
        '<div class="formula-rhs">' + expr + '</div>' +
      '</div>';
    }).join('');
  } else {
    $formulaSection.classList.add('hidden');
  }

  $breakdownGrid.innerHTML = result.breakdown.map(function (item) {
    var tip = item.tip ? ' ' + tipIcon(item.tip) : '';
    return '<div class="breakdown-row">' +
      '<span class="label">' + item.label + tip + '</span>' +
      '<span class="val">' + item.value + '</span>' +
    '</div>';
  }).join('');

  if (formula === 'deepseek_v4_hybrid') {
    $noteSection.textContent = 'Production estimate uses the official sliding-window/compressed-cache layout. The default DeepSeek V4 setting uses FP8 attention cache and FP4 indexer cache.';
  } else {
    $noteSection.textContent = 'Curated from official Hugging Face model config/source files and serving-engine references. Values describe KV cache capacity planning, not model weights or activation memory.';
  }

  $sourceLink.href = model.source_url;
  $sourceLink.textContent = 'Source: ' + model.source_url;
}

buildPicker('');
renderTag();
onModelChange();
