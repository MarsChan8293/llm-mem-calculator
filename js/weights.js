if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

var $modelPicker = document.getElementById('modelPicker');
var $selectedTags = document.getElementById('selectedTags');
var $wtPrecSeg = document.getElementById('wtPrecSeg');
var $totalParams = document.getElementById('totalParams');
var $totalBytes = document.getElementById('totalBytes');
var $metricsCompact = document.getElementById('metricsCompact');
var $formulaSection = document.getElementById('formulaSection');
var $formulaTitle = document.getElementById('formulaTitle');
var $formulaBody = document.getElementById('formulaBody');
var $breakdownGrid = document.getElementById('breakdownGrid');
var $noteSection = document.getElementById('noteSection');
var $sourceLink = document.getElementById('sourceLink');
var $themeToggle = document.getElementById('themeToggle');
var $breakdownToggle = document.getElementById('breakdownToggle');

var selectedModelId = 'deepseek-v4-pro';
var wtPrecValue = 'fp8_int8';

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
  var searchInput = $modelPicker.querySelector('.picker-search');
  var hadFocus = searchInput && document.activeElement === searchInput;
  var cursorPos = searchInput ? searchInput.selectionStart : 0;

  while ($modelPicker.lastChild) {
    $modelPicker.removeChild($modelPicker.lastChild);
  }

  if (!searchInput) {
    searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.className = 'picker-search';
    searchInput.placeholder = 'Search models...';
    searchInput.addEventListener('input', function () {
      buildPicker(searchInput.value);
    });
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
            renderTag();
            buildPicker(searchInput.value);
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
    calculate();
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

function getWtPrecBytes() {
  var opts = MODEL_DATA.weight_precision_options;
  if (!opts) return 2;
  return opts.find(function (p) { return p.id === wtPrecValue; }).bytes_per_element;
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

initSegControl($wtPrecSeg, function (val) {
  wtPrecValue = val;
  calculate();
});

function formatTotal(bytes) {
  return (bytes / 1e9).toFixed(5);
}

function getUnitLabel() {
  return 'GB';
}

function formatMetric(bytes) {
  return (bytes / 1e9).toFixed(5) + ' GB';
}

function formatParams(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + ' B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' K';
  return n.toString();
}

function fmtNum(n) { return n.toLocaleString('en-US'); }

function tipIcon(tooltip) {
  return '<span class="tip-icon" data-tooltip="' + tooltip.replace(/"/g, '&quot;') + '">?</span>';
}

function formatSymbol(text) {
  return text.replace(/^([A-Za-z]+)_(.+)$/, '$1<sub>$2</sub>');
}

function getWeightBarColorClass(type) { return WEIGHT_BAR_COLOR_MAP[type] || 'seg-full'; }
function getWeightBarHex(type) { return WEIGHT_BAR_HEX_MAP[type] || '#4263eb'; }
function getWeightLegendLabel(type) { return WEIGHT_LEGEND_LABEL_MAP[type] || type; }

function calculate() {
  var model = getModel();
  if (!model) {
    $totalParams.textContent = '\u2014';
    $totalBytes.textContent = getUnitLabel();
    $metricsCompact.innerHTML = '';
    $formulaSection.classList.add('hidden');
    $breakdownGrid.innerHTML = '';
    $noteSection.textContent = '';
    $sourceLink.href = '#';
    $sourceLink.textContent = '';
    return;
  }

  var wtPrecB = getWtPrecBytes();
  var result = calcWeight(model, wtPrecB);

  $totalParams.textContent = formatTotal(result.totalBytes);
  $totalBytes.textContent = getUnitLabel();

  var metricsHtml = '';
  metricsHtml += '<span class="metric-item">Attention <span class="metric-val">' + formatMetric(result.attnParams * wtPrecB) + '</span></span>';
  metricsHtml += '<span class="metric-sep">\u00b7</span>';
  metricsHtml += '<span class="metric-item">FFN <span class="metric-val">' + formatMetric((result.ffnDenseParams + result.ffnSharedParams + result.ffnExpertParams) * wtPrecB) + '</span></span>';
  metricsHtml += '<span class="metric-sep">\u00b7</span>';
  metricsHtml += '<span class="metric-item">Embed <span class="metric-val">' + formatMetric(result.embedParams * wtPrecB) + '</span></span>';
  $metricsCompact.innerHTML = metricsHtml;

  if (result.formulas.length > 0) {
    $formulaSection.classList.remove('hidden');
    $formulaTitle.textContent = result.formulaTitle;
    var globalMaxBarBytes = 0;
    result.formulas.forEach(function (f) {
      if (f.bar) {
        f.bar.forEach(function (seg) {
          if (seg.bytes > globalMaxBarBytes) globalMaxBarBytes = seg.bytes;
        });
      }
    });
    var formulaRowsHtml = result.formulas.map(function (f) {
      var expr = f.expr;
      var vals = Object.assign({}, f.values);
      var inputNames = { p_wt: 1 };
      var resultNames = { Attn: 1, Attn_f: 1, Attn_s: 1, Attn_l: 1, Idx: 1, FFN_d: 1, FFN_s: 1, FFN_e: 1, Embed: 1, W_t: 1 };
      var keys = Object.keys(vals).sort(function (a, b) { return b.length - a.length; });
      if (keys.length > 0) {
        var re = new RegExp('\\b(' + keys.map(function (k) { return k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }).join('|') + ')\\b', 'g');
        expr = expr.replace(re, function (match) {
          var val = vals[match];
          var tooltipText;
          if (typeof val === 'number') {
            tooltipText = (WEIGHT_SYMBOL_NAMES[match] || match) + ' = ' + fmtNum(val);
          } else {
            tooltipText = val;
          }
          var cls = 'pill';
          if (inputNames[match]) {
            cls += ' pill-input';
          } else if (resultNames[match]) {
            cls += ' pill-result';
          } else {
            cls += ' pill-param';
          }
          return '<span class="' + cls + '" data-tooltip="' + tooltipText.replace(/"/g, '&quot;') + '">' + formatSymbol(match) + '</span>';
        });
      }
      expr = expr.replace(/\u230a/g, '<span class="floor">\u230a</span>');
      expr = expr.replace(/\u230b/g, '<span class="floor">\u230b</span>');
      var nameVal = f.resultValue !== undefined ? f.resultValue : vals[f.name];
      var nameTooltip = '';
      if (nameVal !== undefined) {
        nameTooltip = typeof nameVal === 'number' ? fmtNum(nameVal) : String(nameVal);
      }
      if (f.tip) {
        nameTooltip = nameTooltip ? nameTooltip + '\n' + f.tip : f.tip;
      }
      var namePill = '<span class="pill pill-result" data-tooltip="' + nameTooltip.replace(/"/g, '&quot;') + '">' + formatSymbol(f.name) + '</span>';

      var ibarHtml = '';
      if (f.bar && f.bar.length > 0) {
        ibarHtml = '<div class="ibar">';
        var formulaBytes = 0;
        f.bar.forEach(function (seg) {
          formulaBytes += seg.bytes;
          var w = globalMaxBarBytes > 0 ? Math.max(1, (seg.bytes / globalMaxBarBytes) * 120) : 1;
          ibarHtml += '<div class="seg ' + getWeightBarColorClass(seg.type) + '" style="width:' + w + 'px" data-tooltip="' + getWeightLegendLabel(seg.type) + ': ' + formatMetric(seg.bytes) + '"></div>';
        });
        ibarHtml += '</div>';
        ibarHtml += '<div class="ibar-val">' + formatMetric(formulaBytes) + '</div>';
      }

      return '<div class="formula-row">' +
        '<div class="formula-lhs">' +
          namePill +
          '<span class="formula-eq">=</span>' +
        '</div>' +
        '<div class="formula-rhs">' + expr + '</div>' +
        ibarHtml +
      '</div>';
    }).join('');
    var patternHtml = '';
    if (result.patterns && result.patterns.length > 0) {
      var maxPatternBytes = 0;
      result.patterns.forEach(function (p) { if (p.bytes > maxPatternBytes) maxPatternBytes = p.bytes; });
      patternHtml = '<div class="pattern-section">';
      patternHtml += '<div class="pattern-title">Layer patterns</div>';
      result.patterns.forEach(function (p) {
        patternHtml += '<div class="pattern-row"><div class="pattern-bar">';
        p.segs.forEach(function (seg) {
          var segBytes = p.bytes * seg.ratio;
          var w = maxPatternBytes > 0 ? Math.max(2, (segBytes / maxPatternBytes) * 120) : 2;
          patternHtml += '<div class="seg ' + getWeightBarColorClass(seg.type) + '" style="width:' + w + 'px" data-tooltip="' + getWeightLegendLabel(seg.type) + ': ' + formatMetric(segBytes) + '"></div>';
        });
        patternHtml += '</div>';
        patternHtml += '<span class="pattern-count">\u00d7' + p.count + '</span>';
        patternHtml += '<span class="pattern-label">' + p.label + '</span>';
        patternHtml += '</div>';
      });
      if (result.legendTypes && result.legendTypes.length >= 1) {
        patternHtml += '<div class="layer-bar-legend">';
        result.legendTypes.forEach(function (t) {
          patternHtml += '<div class="lbleg"><div class="s" style="background:' + getWeightBarHex(t) + '"></div>' + getWeightLegendLabel(t) + '</div>';
        });
        patternHtml += '</div>';
      }
      patternHtml += '</div>';
    }
    $formulaBody.innerHTML = formulaRowsHtml + patternHtml;
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

  $noteSection.textContent = 'Parameter counts derived from official Hugging Face model configs. Excludes LayerNorm, biases, and router weights (<0.1% of total).';
  $sourceLink.href = model.source_url;
  $sourceLink.textContent = 'Source: ' + model.source_url;
}

$breakdownToggle.addEventListener('click', function () {
  var isOpen = $breakdownToggle.classList.toggle('open');
  $breakdownGrid.classList.toggle('collapsed', !isOpen);
});

buildPicker('');
renderTag();
calculate();
