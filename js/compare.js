var PALETTE = [
  '#3b5bdb', '#e03131', '#2b8a3e', '#d9480f', '#9c36b5',
  '#0c8599', '#c2255c', '#e8590c', '#6741d9', '#862e9c'
];

var selectedModels = [];
var chart = null;

var $picker = document.getElementById('modelPicker');
var $tags = document.getElementById('selectedTags');
var $prec = document.getElementById('prec');
var $idxPrec = document.getElementById('idxPrec');
var $seq = document.getElementById('seq');
var $draft = document.getElementById('draft');
var $linear = document.getElementById('linear');
var $draftField = document.getElementById('draftField');
var $linearField = document.getElementById('linearField');
var $idxPrecField = document.getElementById('idxPrecField');
var $chartTitle = document.getElementById('chartTitle');
var $xMax = document.getElementById('xMax');
var $canvas = document.getElementById('chartCanvas');
var $emptyState = document.getElementById('emptyState');
var $btnDownload = document.getElementById('btnDownload');
var $btnCopy = document.getElementById('btnCopy');

var MAX_MODELS = 10;

var _rs = getComputedStyle(document.documentElement);
var CSS_TEXT = _rs.getPropertyValue('--text').trim();
var CSS_TEXT2 = _rs.getPropertyValue('--text2').trim();
var CSS_BORDER = _rs.getPropertyValue('--border').trim();
var CSS_SANS = _rs.getPropertyValue('--sans').trim();

MODEL_DATA.precision_options.forEach(function (p) {
  var opt = document.createElement('option');
  opt.value = p.id;
  opt.textContent = p.label;
  $prec.appendChild(opt);
});
$prec.value = 'fp8_int8';

MODEL_DATA.indexer_precision_options.forEach(function (p) {
  var opt = document.createElement('option');
  opt.value = p.id;
  opt.textContent = p.label;
  $idxPrec.appendChild(opt);
});
$idxPrec.value = 'fp4_int4';

function getPrecBytes() {
  return MODEL_DATA.precision_options.find(function (p) { return p.id === $prec.value; }).bytes_per_element;
}
function getIdxPrecBytes() {
  return MODEL_DATA.indexer_precision_options.find(function (p) { return p.id === $idxPrec.value; }).bytes_per_element;
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
  $picker.innerHTML = '';

  var searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.className = 'picker-search';
  searchInput.placeholder = 'Search models...';
  if (filter) searchInput.value = filter;
  $picker.appendChild(searchInput);

  var query = (filter || '').toLowerCase().trim();

  var selectedIds = {};
  selectedModels.forEach(function (m) { selectedIds[m.id] = true; });
  var atLimit = selectedModels.length >= MAX_MODELS;

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
    $picker.appendChild(header);

    if (!collapsedFamilies[fam]) {
      models.forEach(function (m) {
        var item = document.createElement('div');
        item.className = 'picker-item';
        if (selectedIds[m.id]) item.classList.add('selected');
        if (atLimit && !selectedIds[m.id]) item.classList.add('disabled');

        var nameSpan = document.createElement('span');
        nameSpan.textContent = m.label;

        var addBtn = document.createElement('span');
        addBtn.className = 'picker-add';
        addBtn.textContent = selectedIds[m.id] ? '\u2713' : '+';

        item.appendChild(nameSpan);
        item.appendChild(addBtn);

        if (!atLimit || selectedIds[m.id]) {
          item.addEventListener('click', function () {
            if (selectedIds[m.id]) {
              removeModel(m.id);
            } else {
              addModel(m);
            }
          });
        }

        $picker.appendChild(item);
      });
    }
  });

  searchInput.addEventListener('input', function () {
    buildPicker(searchInput.value);
  });

  searchInput.focus();
}

function addModel(model) {
  if (selectedModels.length >= MAX_MODELS) return;
  if (selectedModels.some(function (m) { return m.id === model.id; })) return;
  selectedModels.push(model);
  refresh();
}

function removeModel(id) {
  selectedModels = selectedModels.filter(function (m) { return m.id !== id; });
  refresh();
}

function renderTags() {
  $tags.innerHTML = '';
  selectedModels.forEach(function (m, idx) {
    var color = PALETTE[idx % PALETTE.length];
    var tag = document.createElement('span');
    tag.className = 'tag';
    tag.style.background = color + '18';
    tag.style.color = color;
    tag.draggable = true;
    tag.dataset.idx = idx;

    var dot = document.createElement('span');
    dot.className = 'tag-dot';
    dot.style.background = color;

    var label = document.createElement('span');
    label.textContent = m.label;

    var removeBtn = document.createElement('span');
    removeBtn.className = 'tag-remove';
    removeBtn.textContent = '\u00d7';
    removeBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      removeModel(m.id);
    });

    tag.appendChild(dot);
    tag.appendChild(label);
    tag.appendChild(removeBtn);

    tag.addEventListener('dragstart', function (e) {
      e.dataTransfer.setData('text/plain', idx.toString());
      e.dataTransfer.effectAllowed = 'move';
      tag.style.opacity = '0.5';
    });
    tag.addEventListener('dragend', function () {
      tag.style.opacity = '1';
    });
    tag.addEventListener('dragover', function (e) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
    });
    tag.addEventListener('drop', function (e) {
      e.preventDefault();
      var fromIdx = parseInt(e.dataTransfer.getData('text/plain'), 10);
      var toIdx = idx;
      if (fromIdx === toIdx) return;
      var moved = selectedModels.splice(fromIdx, 1)[0];
      selectedModels.splice(toIdx, 0, moved);
      refresh();
    });

    $tags.appendChild(tag);
  });
}

function updateConditionalFields() {
  var hasIndexer = selectedModels.some(function (m) {
    return ['deepseek_v4_hybrid', 'dsa_mla'].includes(m.formula);
  });
  $idxPrecField.classList.toggle('hidden', !hasIndexer);

  var hasDraft = selectedModels.some(function (m) {
    var f = m.fields;
    return f.num_nextn_predict_layers || f.mtp_transformer_layers;
  });
  $draftField.classList.toggle('hidden', !hasDraft);

  var hasLinear = selectedModels.some(function (m) {
    return m.formula === 'qwen_linear_full_hybrid';
  });
  $linearField.classList.toggle('hidden', !hasLinear);

  if (!navigator.clipboard || !navigator.clipboard.write) {
    $btnCopy.disabled = true;
  }
}

function generateTokenPoints(maxTokens) {
  var count = 20;
  var points = [];
  for (var i = 0; i <= count; i++) {
    points.push(Math.round(maxTokens * i / count));
  }
  return points;
}

function formatXTick(value) {
  if (value >= 1000000) return (value / 1000000).toFixed(value % 1000000 === 0 ? 0 : 1) + 'M';
  if (value >= 1000) return (value / 1000).toFixed(value % 1000 === 0 ? 0 : 1) + 'K';
  return value.toString();
}

function formatYTick(value) {
  return Math.round(value) + ' GB';
}

function renderChart() {
  if (selectedModels.length === 0) {
    $emptyState.style.display = 'flex';
    $canvas.style.display = 'none';
    if (chart) {
      chart.destroy();
      chart = null;
    }
    return;
  }

  $emptyState.style.display = 'none';
  $canvas.style.display = 'block';

  var precB = getPrecBytes();
  var idxB = getIdxPrecBytes();
  var seqs = Math.max(1, parseInt($seq.value) || 1);
  var maxTokens = parseInt($xMax.value) || 1000000;
  var options = {
    includeDraft: $draft.checked,
    includeLinear: $linear.checked
  };
  var tokenPoints = generateTokenPoints(maxTokens);

  var datasets = selectedModels.map(function (m, idx) {
    var color = PALETTE[idx % PALETTE.length];
    var series = calcKvCacheSeries(m, tokenPoints, precB, idxB, options);
    return {
      label: m.label,
      data: series.map(function (pt) { return { x: pt.tokens, y: pt.gb * seqs }; }),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0
    };
  });

  var config = {
    type: 'line',
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        title: {
          display: true,
          text: $chartTitle.value || 'KV Cache Comparison',
          align: 'center',
          font: { size: 16, weight: 'bold', family: CSS_SANS },
          color: CSS_TEXT,
          padding: { bottom: 16 }
        },
        legend: {
          position: 'top',
          align: 'start',
          labels: {
            usePointStyle: true,
            pointStyle: 'line',
            font: { size: 12, family: CSS_SANS },
            color: CSS_TEXT,
            padding: 16
          }
        },
        tooltip: {
          callbacks: {
            title: function (items) {
              return items[0].dataset.label;
            },
            label: function (ctx) {
              return formatXTick(ctx.parsed.x) + ' tokens \u2192 ' + ctx.parsed.y.toFixed(2) + ' GB';
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: maxTokens,
          title: { display: true, text: 'Tokens', font: { size: 12, family: CSS_SANS }, color: CSS_TEXT2 },
          grid: { color: CSS_BORDER, borderDash: [4, 4], drawTicks: false },
          border: { display: true, color: CSS_BORDER },
          ticks: {
            callback: formatXTick,
            font: { size: 11, family: CSS_SANS },
            color: CSS_TEXT2,
            maxTicksLimit: 10
          }
        },
        y: {
          min: 0,
          title: { display: true, text: 'Cache size (GB)', font: { size: 12, family: CSS_SANS }, color: CSS_TEXT2 },
          grid: { color: CSS_BORDER, borderDash: [4, 4], drawTicks: false },
          border: { display: true, color: CSS_BORDER },
          ticks: {
            callback: formatYTick,
            font: { size: 11, family: CSS_SANS },
            color: CSS_TEXT2
          }
        }
      },
      layout: {
        padding: { top: 4, right: 8, bottom: 0, left: 0 }
      }
    }
  };

  config.options.scales.x.border.display = true;
  config.options.scales.y.border.display = true;

  if (chart) {
    chart.destroy();
  }
  chart = new Chart($canvas, config);
}

function refresh() {
  renderTags();
  buildPicker($picker.querySelector('.picker-search')
    ? $picker.querySelector('.picker-search').value
    : '');
  updateConditionalFields();
  renderChart();
}

function showToast(msg, isError) {
  var existing = document.querySelector('.cmp-toast');
  if (existing) existing.remove();

  var toast = document.createElement('div');
  toast.className = 'cmp-toast ' + (isError ? 'cmp-toast--error' : 'cmp-toast--success');
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(function () { toast.style.opacity = '0'; }, 1800);
  setTimeout(function () { toast.remove(); }, 2200);
}

$btnDownload.addEventListener('click', function () {
  if (!chart) return;
  var link = document.createElement('a');
  link.download = 'kv-cache-comparison.png';
  link.href = chart.toBase64Image('image/png', 1.0);
  link.click();
});

$btnCopy.addEventListener('click', function () {
  if (!chart) return;
  $canvas.toBlob(function (blob) {
    if (!blob) { showToast('Failed to copy', true); return; }
    navigator.clipboard.write([
      new ClipboardItem({ 'image/png': blob })
    ]).then(function () {
      showToast('Copied to clipboard');
    }).catch(function () {
      showToast('Failed to copy', true);
    });
  }, 'image/png');
});

[$prec, $idxPrec, $seq, $chartTitle, $xMax].forEach(function (el) {
  el.addEventListener('change', renderChart);
  el.addEventListener('input', renderChart);
});
$draft.addEventListener('change', renderChart);
$linear.addEventListener('change', renderChart);

refresh();
