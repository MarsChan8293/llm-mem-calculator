/**
 * calculator.js — DOM interaction logic for the KV Cache Calculator page.
 *
 * Depends on globals:
 *   MODEL_DATA   (from data.js)
 *   calcKvCache, fmtBytes, fmtNum  (from calc.js)
 */

// ═══════════════════════════════════════════════════════════════
// DOM refs
// ═══════════════════════════════════════════════════════════════
var $family = document.getElementById('family');
var $model  = document.getElementById('model');
var $tokens = document.getElementById('tokens');
var $seq    = document.getElementById('seq');
var $prec   = document.getElementById('prec');
var $idxPrec = document.getElementById('idxPrec');
var $draft  = document.getElementById('draft');
var $linear = document.getElementById('linear');
var $idxPrecField = document.getElementById('idxPrecField');
var $draftField  = document.getElementById('draftField');
var $linearField = document.getElementById('linearField');
var $draftHint   = document.getElementById('draftHint');

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════
function tipIcon(tooltip) {
  return '<span class="tip-icon" data-tooltip="' + tooltip.replace(/"/g, '&quot;') + '">?</span>';
}

// ═══════════════════════════════════════════════════════════════
// Init dropdowns
// ═══════════════════════════════════════════════════════════════
var families = [...new Set(MODEL_DATA.models.map(function (m) { return m.family; }))];
families.forEach(function (f) {
  var opt = document.createElement('option');
  opt.value = f; opt.textContent = f;
  $family.appendChild(opt);
});

MODEL_DATA.precision_options.forEach(function (p) {
  var opt = document.createElement('option');
  opt.value = p.id; opt.textContent = p.label;
  $prec.appendChild(opt);
});
$prec.value = 'fp8_int8';

MODEL_DATA.indexer_precision_options.forEach(function (p) {
  var opt = document.createElement('option');
  opt.value = p.id; opt.textContent = p.label;
  $idxPrec.appendChild(opt);
});
$idxPrec.value = 'fp4_int4';

function populateModels(family) {
  $model.innerHTML = '';
  var models = MODEL_DATA.models.filter(function (m) { return m.family === family; });
  models.forEach(function (m) {
    var opt = document.createElement('option');
    opt.value = m.id; opt.textContent = m.label;
    $model.appendChild(opt);
  });
}
populateModels($family.value);

function getModel() {
  return MODEL_DATA.models.find(function (m) { return m.id === $model.value; });
}

function getPrecBytes() {
  return MODEL_DATA.precision_options.find(function (p) { return p.id === $prec.value; }).bytes_per_element;
}

function getIdxPrecBytes() {
  return MODEL_DATA.indexer_precision_options.find(function (p) { return p.id === $idxPrec.value; }).bytes_per_element;
}

// ═══════════════════════════════════════════════════════════════
// Calculate & Render
// ═══════════════════════════════════════════════════════════════
function calculate() {
  var model = getModel();
  if (!model) return;

  var tokens = Math.max(1, parseInt($tokens.value) || 1);
  var seqs   = Math.max(1, parseInt($seq.value) || 1);
  var precB  = getPrecBytes();
  var idxB   = getIdxPrecBytes();
  var formula = model.formula;
  var f = model.fields;
  var includeDraft  = $draft.checked;
  var includeLinear = $linear.checked;

  // Show/hide conditional fields
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

  // Run calculation via calc.js
  var result = calcKvCache(model, tokens, precB, idxB, { includeDraft: includeDraft, includeLinear: includeLinear });

  // Compute total with sequences
  var totalBytes = seqs * (result.kvBytes + result.idxBytes);

  // Fix breakdown "Total bytes" to include sequences
  var lastIdx = result.breakdown.length - 1;
  if (lastIdx >= 0 && result.breakdown[lastIdx].label === 'Total bytes') {
    result.breakdown[lastIdx].value = fmtNum(totalBytes);
  }

  // ── Render metrics ──
  document.getElementById('totalGiB').textContent = (totalBytes / 1024**3).toFixed(5) + ' GiB';
  document.getElementById('totalGB').textContent = '= ' + (totalBytes / 1e9).toFixed(5) + ' GB';
  document.getElementById('kvSize').textContent = fmtBytes(result.kvBytes);
  document.getElementById('idxSize').textContent = result.idxBytes > 0 ? fmtBytes(result.idxBytes) : '—';
  document.getElementById('perToken').textContent = fmtBytes(result.perTokenBytes);

  // ── Render formulas ──
  var $formulaSection = document.getElementById('formulaSection');
  var $formulaTitle = document.getElementById('formulaTitle');
  var $formulaBody = document.getElementById('formulaBody');

  if (result.formulas.length > 0) {
    $formulaSection.classList.remove('hidden');
    $formulaTitle.textContent = result.formulaTitle;
    $formulaBody.innerHTML = result.formulas.map(function (f) {
      return '<div class="formula-row">' +
        '<span class="formula-name">' + f.name + ' ' + tipIcon(f.tip) + '</span>' +
        '<span class="formula-eq">=</span>' +
        '<span class="formula-expr">' + f.expr + '</span>' +
      '</div>';
    }).join('');
  } else {
    $formulaSection.classList.add('hidden');
  }

  // ── Render breakdown ──
  var $grid = document.getElementById('breakdownGrid');
  $grid.innerHTML = result.breakdown.map(function (item) {
    var tip = item.tip ? ' ' + tipIcon(item.tip) : '';
    return '<div class="breakdown-row">' +
      '<span class="label">' + item.label + tip + '</span>' +
      '<span class="val">' + item.value + '</span>' +
    '</div>';
  }).join('');

  // ── Note & source ──
  var noteEl = document.getElementById('noteSection');
  if (formula === 'deepseek_v4_hybrid') {
    noteEl.textContent = 'Production estimate uses the official sliding-window/compressed-cache layout. The default DeepSeek V4 setting uses FP8 attention cache and FP4 indexer cache.';
  } else {
    noteEl.textContent = 'Curated from official Hugging Face model config/source files and serving-engine references. Values describe KV cache capacity planning, not model weights or activation memory.';
  }

  var srcLink = document.getElementById('sourceLink');
  srcLink.href = model.source_url;
  srcLink.textContent = 'Source: ' + model.source_url;
}

// ═══════════════════════════════════════════════════════════════
// Event listeners
// ═══════════════════════════════════════════════════════════════
$family.addEventListener('change', function () {
  populateModels($family.value);
  calculate();
});
$model.addEventListener('change', calculate);
$tokens.addEventListener('input', calculate);
$seq.addEventListener('input', calculate);
$prec.addEventListener('change', calculate);
$idxPrec.addEventListener('change', calculate);
$draft.addEventListener('change', calculate);
$linear.addEventListener('change', calculate);

// Initial calc
calculate();
