/**
 * calc.js — Pure KV-cache calculation functions.
 *
 * No DOM dependencies. Relies on the global `MODEL_DATA` (or `DATA`) object
 * being available before this script is loaded.
 *
 * Exports (global):
 *   fmtBytes, fmtBytesGB, fmtNum   — utility formatters
 *   calcKvCache                    — main calculation, returns full result object
 *   calcKvCacheGB                  — convenience, returns total GB
 *   calcKvCacheSeries              — returns [{tokens, gb}, …] for chart data
 */

/**
 * Mapping from short math symbols to human-readable field names.
 * Used by calculator.js to build tooltips: "{fullName} = {value}".
 */
var SYMBOL_NAMES = {
  L: 'layers', T: 'tokens', B: 'batch_size', p: 'precision', p_idx: 'indexer precision',
  h_kv: 'kv_heads', d_h: 'head_dim',
  d_c: 'kv_lora_rank', d_r: 'qk_rope_head_dim', d_idx: 'index_head_dim',
  W: 'sliding_window', r: 'compress_ratio',
  L_4: 'ratio4_layers', L_128: 'ratio128_layers', L_0: 'ratio0_layers',
  L_f: 'full_attention_layers', L_s: 'sliding_attention_layers', L_l: 'linear_attention_layers',
  h_f: 'full_kv_heads', h_s: 'sliding_kv_heads',
  d_f: 'full_head_dim', d_s: 'sliding_head_dim',
  d_vf: 'full_v_head_dim', d_vs: 'sliding_v_head_dim',
  h_kl: 'linear_key_heads', h_vl: 'linear_value_heads',
  d_kl: 'linear_key_head_dim', d_vl: 'linear_value_head_dim',
  k_c: 'conv_kernel_dim',
  h_idx: 'sparse_index_heads', L_sp: 'sparse_layers',
  L_idx: 'indexer_source_layers', k_pool: 'indexer_pool_size',
  r_idx: 'indexer_compress_ratio', h_idx_kv: 'indexer_kv_heads',
  h_l: 'linear_num_heads', d_l: 'linear_head_dim', qkv: 'linear_qkv_width',
  s_idx: 'indexer_scale_bytes_per_pool'
};

/**
 * Format bytes as decimal GB so every calculated memory value shares one unit.
 * @param {number} bytes
 * @returns {string}
 */
function fmtBytes(bytes) {
  return (bytes / 1e9).toFixed(5) + ' GB';
}

/**
 * Format bytes using decimal (SI) prefixes (KB, MB, GB).
 * @param {number} bytes
 * @returns {string}
 */
function fmtBytesGB(bytes) {
  if (bytes < 1e3) return bytes.toFixed(0) + ' B';
  if (bytes < 1e6) return (bytes / 1e3).toFixed(5) + ' KB';
  if (bytes < 1e9) return (bytes / 1e6).toFixed(5) + ' MB';
  return (bytes / 1e9).toFixed(5) + ' GB';
}

/**
 * Format a number with locale-aware thousand separators.
 * @param {number} n
 * @returns {string}
 */
function fmtNum(n) { return n.toLocaleString('en-US'); }

/**
 * Calculate KV-cache size for a given model configuration.
 *
 * @param {Object} model        - Model entry from MODEL_DATA (has .fields, .formula, .label)
 * @param {number} tokens       - Sequence length (number of tokens)
 * @param {number} precB        - Bytes per element for KV precision
 * @param {number} idxB         - Bytes per element for indexer precision
 * @param {Object} options      - { includeDraft: bool, includeLinear: bool }
 * @returns {{ kvBytes: number, idxBytes: number, mainKvBytes: number, draftKvBytes: number,
 *             kvLayerBytes: Array<number>|null, idxLayerBytes: Array<number>|null,
 *             perTokenBytes: number,
 *             breakdown: Array<{label:string,value:string,tip?:string}>,
 *             formulas: Array<{name:string,tip:string,expr:string}>,
 *             formulaTitle: string }}
 */
function calcKvCache(model, tokens, precB, idxB, options) {
  const f = model.fields;
  const formula = model.formula;
  const includeDraft  = !!(options && options.includeDraft);
  const includeLinear = !!(options && options.includeLinear);
  var seqs = (options && options.seqs) || 1;

  let kvBytes = 0, idxBytes = 0, perTokenBytes = 0;
  let breakdown = [];
  let formulas = [];
  let formulaTitle = '';
  var patterns = [];
  var idxLayers = 0;
  var legendTypes = [];
  // New hybrid formulas expose ownership of persistent cache state so the
  // deployment calculator can place shared source caches on the right PP
  // stage instead of spreading them uniformly across all layers.
  var mainKvBytes = null;
  var draftKvBytes = 0;
  var kvLayerBytes = null;
  var idxLayerBytes = null;
  var linearLayerBytes = null;
  var globalCacheBytes = null;

  // ── standard_gqa ──
  if (formula === 'standard_gqa') {
    const layers = f.num_hidden_layers;
    const kvHeads = f.num_key_value_heads;
    const hd = f.head_dim;
    const elements = 2 * layers * kvHeads * hd * tokens;
    kvBytes = elements * precB;
    perTokenBytes = 2 * layers * kvHeads * hd * precB;

    let draftLayers = 0;
    if (includeDraft && f.mtp_transformer_layers) {
      draftLayers = f.mtp_transformer_layers;
      const draftElements = 2 * draftLayers * kvHeads * hd * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += 2 * draftLayers * kvHeads * hd * precB;
    }

    formulaTitle = model.label + ' standard GQA';
    formulas = [
      { name: 'KV', tip: 'Main ' + model.label + ' KV cache before applying precision.', expr: '2 \u00d7 L \u00d7 h_kv \u00d7 d_h \u00d7 T \u00d7 p', values: { L: layers, h_kv: kvHeads, d_h: hd, T: tokens, p: precB }, resultValue: kvBytes, bar: [{ type: 'full', bytes: kvBytes }], ibarVal: fmtBytes(kvBytes) },
      { name: 'Total', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'B \u00d7 KV', values: { KV: kvBytes }, resultValue: kvBytes, bar: [{ type: 'full', bytes: kvBytes }], ibarVal: fmtBytes(seqs * kvBytes) }
    ];
    patterns = [{
      segs: [{ type: 'full', ratio: 0.5 }, { type: 'full-alt', ratio: 0.5 }],
      count: layers,
      label: 'all layers',
      bytes: kvBytes / layers
    }];
    legendTypes = ['full'];

    breakdown = [
      { label: 'Layers', value: fmtNum(layers) },
      { label: 'KV heads', value: fmtNum(kvHeads) },
      { label: 'Head dim', value: fmtNum(hd) },
      { label: 'KV elements', value: fmtNum(elements) },
      { label: 'KV precision bytes', value: precB.toString() },
    ];
    if (draftLayers > 0) {
      breakdown.push({ label: 'Draft layers included', value: fmtNum(draftLayers), tip: 'Extra MTP/draft layers after the main transformer layers.' });
    }
    breakdown.push({ label: 'Total bytes', value: fmtNum(kvBytes) });

  // ── mla ──
  } else if (formula === 'mla') {
    const layers = f.num_hidden_layers;
    const kvLoraRank = f.kv_lora_rank;
    const qkRopeHd = f.qk_rope_head_dim || 0;
    // MLA: KV cache stores a single compressed latent (kv_lora_rank) plus a RoPE key (qk_rope_head_dim)
    const elements = layers * (kvLoraRank + qkRopeHd) * tokens;
    kvBytes = elements * precB;
    perTokenBytes = layers * (kvLoraRank + qkRopeHd) * precB;

    let draftLayers = 0;
    if (includeDraft && f.num_nextn_predict_layers) {
      draftLayers = f.num_nextn_predict_layers;
      const draftElements = draftLayers * (kvLoraRank + qkRopeHd) * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += draftLayers * (kvLoraRank + qkRopeHd) * precB;
    }

    formulaTitle = model.label + ' multi-head latent attention (MLA)';
    var mlaCompressedBytes = kvBytes * kvLoraRank / (kvLoraRank + qkRopeHd);
    var mlaRopeBytes = kvBytes * qkRopeHd / (kvLoraRank + qkRopeHd);
    var mlaKvBar = qkRopeHd > 0
      ? [{ type: 'compressed', bytes: mlaCompressedBytes }, { type: 'rope', bytes: mlaRopeBytes }]
      : [{ type: 'compressed', bytes: kvBytes }];
    formulas = [
      { name: 'KV', tip: 'Compressed KV cache using MLA projection. Each layer stores (kv_lora_rank + qk_rope_head_dim) elements per token.', expr: 'L \u00d7 (d_c + d_r) \u00d7 T \u00d7 p', values: { L: layers, d_c: kvLoraRank, d_r: qkRopeHd, T: tokens, p: precB }, resultValue: kvBytes, bar: mlaKvBar, ibarVal: fmtBytes(kvBytes) },
      { name: 'Total', tip: 'Combined cache payload for all concurrent sequences.', expr: 'B \u00d7 KV', values: { KV: kvBytes }, resultValue: kvBytes, bar: mlaKvBar, ibarVal: fmtBytes(seqs * kvBytes) }
    ];
    var mlaDenom = kvLoraRank + qkRopeHd;
    patterns = [{
      segs: mlaDenom > 0
        ? [{ type: 'compressed', ratio: kvLoraRank / mlaDenom }, { type: 'rope', ratio: qkRopeHd / mlaDenom }]
        : [{ type: 'compressed', ratio: 1 }],
      count: layers,
      label: 'all layers',
      bytes: kvBytes / layers
    }];
    legendTypes = qkRopeHd > 0 ? ['compressed', 'rope'] : ['compressed'];

    breakdown = [
      { label: 'Layers', value: fmtNum(layers) },
      { label: 'KV LoRA rank', value: fmtNum(kvLoraRank) },
      { label: 'QK RoPE head dim', value: fmtNum(qkRopeHd) },
      { label: 'KV elements', value: fmtNum(elements) },
      { label: 'KV precision bytes', value: precB.toString() },
    ];
    if (draftLayers > 0) {
      breakdown.push({ label: 'Draft layers included', value: fmtNum(draftLayers), tip: 'Extra MTP/draft layers after the main transformer layers.' });
    }
    breakdown.push({ label: 'Total bytes', value: fmtNum(kvBytes) });

  // ── dsa_mla ──
  } else if (formula === 'dsa_mla') {
    const layers = f.num_hidden_layers;
    const kvLoraRank = f.kv_lora_rank;
    const qkRopeHd = f.qk_rope_head_dim || 0;
    const elements = layers * (kvLoraRank + qkRopeHd) * tokens;
    kvBytes = elements * precB;
    perTokenBytes = layers * (kvLoraRank + qkRopeHd) * precB;

    const idxHd = f.index_head_dim;
    // IndexShare: every `index_topk_freq` layers share one indexer.
    // Only "full" indexer layers store indexer key cache; "shared" layers reuse topk indices.
    var numIndexerLayers = layers;
    var indexShareFreq = f.index_topk_freq || 0;
    if (indexShareFreq > 0) {
      var indexShareOffset = f.index_skip_topk_offset || 0;
      numIndexerLayers = indexShareOffset + Math.floor((layers - indexShareOffset) / indexShareFreq);
    }
    const idxElements = numIndexerLayers * idxHd * tokens;
    idxBytes = idxElements * idxB;
    perTokenBytes += numIndexerLayers * idxHd * idxB;
    idxLayers = numIndexerLayers;

    let draftLayers = 0;
    if (includeDraft && f.num_nextn_predict_layers) {
      draftLayers = f.num_nextn_predict_layers;
      const draftElements = draftLayers * (kvLoraRank + qkRopeHd) * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += draftLayers * (kvLoraRank + qkRopeHd) * precB;
    }

    formulaTitle = model.label + ' DSA + MLA attention';
    var dsaCompressedBytes = kvBytes * kvLoraRank / (kvLoraRank + qkRopeHd);
    var dsaRopeBytes = kvBytes * qkRopeHd / (kvLoraRank + qkRopeHd);
    var dsaKvBar = qkRopeHd > 0
      ? [{ type: 'compressed', bytes: dsaCompressedBytes }, { type: 'rope', bytes: dsaRopeBytes }]
      : [{ type: 'compressed', bytes: kvBytes }];
    var dsaTotalBar = qkRopeHd > 0
      ? [{ type: 'compressed', bytes: dsaCompressedBytes }, { type: 'rope', bytes: dsaRopeBytes }, { type: 'indexer', bytes: idxBytes }]
      : [{ type: 'compressed', bytes: kvBytes }, { type: 'indexer', bytes: idxBytes }];
    formulas = [
      { name: 'KV', tip: 'Compressed KV cache using MLA projection. Each layer stores (kv_lora_rank + qk_rope_head_dim) elements per token.', expr: 'L \u00d7 (d_c + d_r) \u00d7 T \u00d7 p', values: { L: layers, d_c: kvLoraRank, d_r: qkRopeHd, T: tokens, p: precB }, resultValue: kvBytes, bar: dsaKvBar, ibarVal: fmtBytes(kvBytes) },
      { name: 'Idx', tip: indexShareFreq > 0 ? 'Indexer cache for DSA sparse attention lookup. With IndexShare, only ' + numIndexerLayers + ' of ' + layers + ' layers store indexer keys (every ' + indexShareFreq + ' layers share one).' : 'Indexer cache for DSA sparse attention lookup.', expr: indexShareFreq > 0 ? 'L_idx \u00d7 d_idx \u00d7 T \u00d7 p_idx' : 'L \u00d7 d_idx \u00d7 T \u00d7 p_idx', values: indexShareFreq > 0 ? { L_idx: numIndexerLayers, d_idx: idxHd, T: tokens, p_idx: idxB } : { L: layers, d_idx: idxHd, T: tokens, p_idx: idxB }, resultValue: idxBytes, bar: [{ type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(idxBytes) },
      { name: 'Total', tip: 'Combined cache payload for all concurrent sequences.', expr: 'B \u00d7 (KV + Idx)', values: { KV: kvBytes, Idx: idxBytes }, resultValue: kvBytes + idxBytes, bar: dsaTotalBar, ibarVal: fmtBytes(seqs * (kvBytes + idxBytes)) }
    ];
    if (indexShareFreq > 0) {
      var fullDenom = kvLoraRank + qkRopeHd + idxHd;
      var sharedDenom = kvLoraRank + qkRopeHd;
      patterns = [
        { segs: qkRopeHd > 0
            ? [{ type: 'compressed', ratio: kvLoraRank / fullDenom }, { type: 'rope', ratio: qkRopeHd / fullDenom }, { type: 'indexer', ratio: idxHd / fullDenom }]
            : [{ type: 'compressed', ratio: kvLoraRank / fullDenom }, { type: 'indexer', ratio: idxHd / fullDenom }],
          count: numIndexerLayers,
          label: 'indexer layers',
          bytes: ((kvLoraRank + qkRopeHd) * precB + idxHd * idxB) * tokens
        },
        { segs: qkRopeHd > 0
            ? [{ type: 'compressed', ratio: kvLoraRank / sharedDenom }, { type: 'rope', ratio: qkRopeHd / sharedDenom }]
            : [{ type: 'compressed', ratio: 1 }],
          count: layers - numIndexerLayers,
          label: 'shared layers (no indexer)',
          bytes: (kvLoraRank + qkRopeHd) * precB * tokens
        }
      ];
    } else {
      var dsaDenom = kvLoraRank + qkRopeHd + idxHd;
      patterns = [{
        segs: qkRopeHd > 0
          ? [{ type: 'compressed', ratio: kvLoraRank / dsaDenom }, { type: 'rope', ratio: qkRopeHd / dsaDenom }, { type: 'indexer', ratio: idxHd / dsaDenom }]
          : [{ type: 'compressed', ratio: kvLoraRank / dsaDenom }, { type: 'indexer', ratio: idxHd / dsaDenom }],
        count: layers,
        label: 'all layers',
        bytes: (kvBytes + idxBytes) / layers
      }];
    }
    legendTypes = qkRopeHd > 0 ? ['compressed', 'rope', 'indexer'] : ['compressed', 'indexer'];

    breakdown = [
      { label: 'Layers', value: fmtNum(layers) },
      { label: 'KV LoRA rank', value: fmtNum(kvLoraRank) },
      { label: 'QK RoPE head dim', value: fmtNum(qkRopeHd) },
      { label: 'KV elements', value: fmtNum(elements) },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'Indexer head dim', value: fmtNum(idxHd) },
    ];
    if (indexShareFreq > 0) {
      breakdown.push({ label: 'Indexer layers (IndexShare)', value: fmtNum(numIndexerLayers), tip: 'Only ' + numIndexerLayers + ' of ' + layers + ' layers store indexer keys. Every ' + indexShareFreq + ' layers share one indexer.' });
    }
    breakdown.push({ label: 'Indexer elements', value: fmtNum(idxElements) });
    breakdown.push({ label: 'Indexer precision bytes', value: idxB.toString() });
    if (draftLayers > 0) {
      breakdown.push({ label: 'Draft layers included', value: fmtNum(draftLayers), tip: 'Extra MTP/draft layers after the main transformer layers.' });
    }
    breakdown.push({ label: 'Total bytes', value: fmtNum(kvBytes + idxBytes) });

  // ── deepseek_v4_hybrid ──
  } else if (formula === 'deepseek_v4_hybrid') {
    const ratios = f.compress_ratios;
    const sw = f.sliding_window;
    const hd = f.head_dim;
    const idxHd = f.index_head_dim;
    const totalLayers = f.num_hidden_layers;

    // Count layers by ratio
    const ratioCounts = {};
    ratios.forEach(function (r) { ratioCounts[r] = (ratioCounts[r] || 0) + 1; });
    const activeLayers = ratios.filter(function (r) { return r > 0; }).length;
    const ratio0Layers = ratioCounts[0] || 0;
    const ratio4Layers = ratioCounts[4] || 0;
    const ratio128Layers = ratioCounts[128] || 0;

    // Sliding window KV: ALL layers contribute to the sliding window
    const slidingElements = totalLayers * sw * hd;
    // Compressed KV: sum over ratio>0 layers of floor(tokens/ratio) * hd
    var compressedElements = 0;
    ratios.forEach(function (r) {
      if (r > 0) compressedElements += Math.floor(tokens / r) * hd;
    });
    const kvElements = slidingElements + compressedElements;
    kvBytes = kvElements * precB;

    // Ratio=0 layers: only sliding window
    const ratio0Elements = ratio0Layers * sw * hd;

    // Indexer: ratio=4 layers
    const idxElements = ratio4Layers * Math.floor(tokens / 4) * idxHd;
    idxBytes = idxElements * idxB;

    perTokenBytes = (kvElements / tokens) * precB + (idxElements / tokens) * idxB;

    var draftKvBytes = 0;
    var draftLayers = 0;
    if (includeDraft) {
      draftLayers = ratio0Layers;
      draftKvBytes = ratio0Elements * precB;
    }

    const totalKvBytes = kvBytes + draftKvBytes;

    formulaTitle = model.label + ' hybrid sparse attention';
    var v4WindowPerLayer = sw * hd * precB;
    var v4CompressR4PerLayer = Math.floor(tokens / 4) * hd * precB;
    var v4CompressR128PerLayer = Math.floor(tokens / 128) * hd * precB;
    var v4IdxR4PerLayer = Math.floor(tokens / 4) * idxHd * idxB;
    // Total bar window segment includes draft contribution
    var v4TotalWindowBytes = totalKvBytes - compressedElements * precB;
    var v4CompressR4Bytes = ratio4Layers * v4CompressR4PerLayer;
    var v4CompressR128Bytes = ratio128Layers * v4CompressR128PerLayer;
    formulas = [
      { name: 'KV_sw', tip: 'ALL layers contribute to the sliding window KV, including ratio=0 layers.', expr: 'L \u00d7 W \u00d7 d_h \u00d7 p', values: { L: totalLayers, W: sw, d_h: hd, p: precB }, resultValue: slidingElements * precB, bar: [{ type: 'window', bytes: slidingElements * precB }], ibarVal: fmtBytes(slidingElements * precB) },
      { name: 'KV_r4', tip: 'Compressed KV from ratio=4 layers; each layer keeps floor(T/4) compressed slots.', expr: 'L_4 \u00d7 \u230aT/4\u230b \u00d7 d_h \u00d7 p', values: { L_4: ratio4Layers, T: tokens, d_h: hd, p: precB }, resultValue: v4CompressR4Bytes, bar: [{ type: 'compressed', bytes: v4CompressR4Bytes }], ibarVal: fmtBytes(v4CompressR4Bytes) },
      { name: 'KV_r128', tip: 'Compressed KV from ratio=128 layers; each layer keeps floor(T/128) compressed slots.', expr: 'L_128 \u00d7 \u230aT/128\u230b \u00d7 d_h \u00d7 p', values: { L_128: ratio128Layers, T: tokens, d_h: hd, p: precB }, resultValue: v4CompressR128Bytes, bar: [{ type: 'compressed', bytes: v4CompressR128Bytes }], ibarVal: fmtBytes(v4CompressR128Bytes) },
      { name: 'KV_cmp', tip: 'Total compressed KV cache from all layers with compress_ratio > 0.', expr: 'KV_r4 + KV_r128', values: { KV_r4: v4CompressR4Bytes, KV_r128: v4CompressR128Bytes }, resultValue: compressedElements * precB, bar: [{ type: 'compressed', bytes: v4CompressR4Bytes }, { type: 'compressed', bytes: v4CompressR128Bytes }], ibarVal: fmtBytes(compressedElements * precB) },
      { name: 'KV', tip: 'Main ' + model.label + ' KV cache before adding the separate indexer cache.', expr: 'KV_sw + KV_cmp', values: { KV_sw: kvBytes, KV_cmp: compressedElements * precB }, resultValue: kvBytes, bar: [{ type: 'window', bytes: slidingElements * precB }, { type: 'compressed', bytes: compressedElements * precB }], ibarVal: fmtBytes(kvBytes) },
      { name: 'Idx', tip: 'Ratio=4 layers keep an extra compressed indexer cache that can use a separate precision.', expr: 'L_4 \u00d7 \u230aT/4\u230b \u00d7 d_idx \u00d7 p_idx', values: { L_4: ratio4Layers, T: tokens, d_idx: idxHd, p_idx: idxB }, resultValue: idxBytes, bar: [{ type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(idxBytes) },
      { name: 'Total', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'B \u00d7 (KV + Idx)', values: { KV: totalKvBytes, Idx: idxBytes }, resultValue: totalKvBytes + idxBytes, bar: [{ type: 'window', bytes: v4TotalWindowBytes }, { type: 'compressed', bytes: compressedElements * precB }, { type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(seqs * (totalKvBytes + idxBytes)) }
    ];
    var v4R4Total = v4WindowPerLayer + v4CompressR4PerLayer + v4IdxR4PerLayer;
    var v4R128Total = v4WindowPerLayer + v4CompressR128PerLayer;
    patterns = [];
    if (ratio4Layers > 0) {
      patterns.push({
        segs: [
          { type: 'window', ratio: v4WindowPerLayer / v4R4Total },
          { type: 'compressed', ratio: v4CompressR4PerLayer / v4R4Total },
          { type: 'indexer', ratio: v4IdxR4PerLayer / v4R4Total }
        ],
        count: ratio4Layers,
        label: 'r = 4',
        bytes: v4R4Total
      });
    }
    if (ratio128Layers > 0) {
      patterns.push({
        segs: [
          { type: 'window', ratio: v4WindowPerLayer / v4R128Total },
          { type: 'compressed', ratio: v4CompressR128PerLayer / v4R128Total }
        ],
        count: ratio128Layers,
        label: 'r = 128',
        bytes: v4R128Total
      });
    }
    if (ratio0Layers > 0) {
      patterns.push({
        segs: [{ type: 'window-empty', ratio: 1 }],
        count: ratio0Layers,
        label: 'r = 0',
        bytes: 0
      });
    }
    legendTypes = ['window', 'compressed', 'indexer'];

    breakdown = [
      { label: 'Main layers', value: fmtNum(totalLayers) },
      { label: 'Draft layers included', value: includeDraft ? fmtNum(draftLayers) : '0', tip: 'Extra MTP/draft layers after the main transformer layers. In DeepSeek V4 configs these are ratio=0 layers.' },
      { label: 'Ratio=4 layers', value: fmtNum(ratio4Layers), tip: 'Layers whose compressed cache ratio is 4; these layers also carry indexer cache.' },
      { label: 'Ratio=128 layers', value: fmtNum(ratio128Layers), tip: 'Layers whose compressed cache keeps floor(tokens / 128) compressed KV slots.' },
      { label: 'Ratio=0 layers', value: fmtNum(ratio0Layers), tip: 'Layers with no compressed KV segment; they keep only the sliding-window KV cache.' },
      { label: 'Ratio=0 KV elements', value: fmtNum(ratio0Elements), tip: 'The ratio=0 contribution: ratio0_layers \u00d7 sliding_window \u00d7 head_dim.' },
      { label: 'Sliding-window elements', value: fmtNum(slidingElements), tip: 'Per-layer local KV reserve: sliding_window \u00d7 head_dim, summed across active layers.' },
      { label: 'Compressed elements', value: fmtNum(compressedElements), tip: 'Compressed KV elements from layers with compress_ratio greater than zero.' },
      { label: 'KV elements', value: fmtNum(kvElements), tip: 'Sliding-window plus compressed attention cache elements before applying KV precision.' },
      { label: 'Indexer elements', value: fmtNum(idxElements), tip: 'Compressed indexer elements from ratio=4 layers before applying indexer precision.' },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'Indexer precision bytes', value: idxB.toString() },
    ];

    // kvBytes excludes draft; totalKvBytes includes draft.
    // Return kvBytes = totalKvBytes so single-sequence KV is complete.
    kvBytes = totalKvBytes;

  // ── deepseek_v41 ──
  } else if (formula === 'deepseek_v41') {
    const totalLayers = f.num_hidden_layers;
    const sw = f.sliding_window;
    const hd = f.head_dim;
    const idxHd = f.index_head_dim;
    const ratios = Array.isArray(f.compress_ratios) ? f.compress_ratios : [];
    const kvSources = Array.isArray(f.kv_source_layer_ids) ? f.kv_source_layer_ids : [];
    const idxSources = Array.isArray(f.index_source_layer_ids) ? f.index_source_layer_ids : [];
    // Reindex layers execute queries but reuse the most recent owner's K.
    // Official inference/model.py: Indexer.owns_k = layer_id in kv_source_layers.
    const idxOwners = kvSources.filter(function (layer) { return idxSources.indexOf(layer) >= 0; });
    // Packed FP4 payload, including one byte of scale per group (not the
    // dequantized BF16 tensors used by the minimal reference implementation).
    const kvSlotBytes = hd * precB + (precB === 0.5 ? Math.ceil(hd / 16) : 0);
    const idxSlotBytes = idxHd * idxB + (idxB === 0.5 ? Math.ceil(idxHd / 32) : 0);

    // V4.1's compressed KV is source-owned and shared by consuming layers.
    // Do not sum one compressed cache for every layer with the same ratio:
    // only the four kv_source_layer_ids allocate these persistent caches.
    const windowElements = totalLayers * sw * hd;
    const windowBytes = windowElements * precB;
    var compressedElementsV41 = 0;
    var compressedBytesV41 = 0;
    var kvSourcesWithCache = 0;
    kvLayerBytes = [];
    for (var v41Layer = 0; v41Layer < totalLayers; v41Layer++) {
      kvLayerBytes[v41Layer] = sw * hd * precB;
    }
    kvSources.forEach(function (sourceLayer) {
      var ratio = ratios[sourceLayer] || 0;
      if (ratio <= 0) return;
      var slots = Math.floor(tokens / ratio);
      var sourceElements = slots * hd;
      var sourceBytes = slots * kvSlotBytes;
      compressedElementsV41 += sourceElements;
      compressedBytesV41 += sourceBytes;
      kvSourcesWithCache++;
      if (sourceLayer >= 0 && sourceLayer < kvLayerBytes.length) kvLayerBytes[sourceLayer] += sourceBytes;
    });
    mainKvBytes = windowBytes + compressedBytesV41;

    var draftLayersV41 = includeDraft ? (f.num_nextn_predict_layers || 0) : 0;
    draftKvBytes = draftLayersV41 * sw * hd * precB;
    kvBytes = mainKvBytes + draftKvBytes;

    // Only K owners allocate persistent keys; query/reindex layers do not.
    var idxElementsV41 = 0;
    idxLayerBytes = [];
    idxOwners.forEach(function (sourceLayer) {
      var ratio = ratios[sourceLayer] || 0;
      if (ratio <= 0) return;
      var slots = Math.floor(tokens / ratio);
      var sourceIdxBytes = slots * idxSlotBytes;
      idxElementsV41 += slots * idxHd;
      idxBytes += sourceIdxBytes;
      if (sourceLayer >= 0 && sourceLayer < totalLayers) idxLayerBytes[sourceLayer] = (idxLayerBytes[sourceLayer] || 0) + sourceIdxBytes;
    });
    idxLayers = idxOwners.length;
    globalCacheBytes = compressedBytesV41 + idxBytes;
    perTokenBytes = (kvBytes + idxBytes) / tokens;

    formulaTitle = model.label + ' CED + shared compressed KV';
    formulas = [
      { name: 'KV_sw', tip: 'All 40 CED backbone layers keep one sliding-window latent vector per retained token.', expr: 'L × W × d_h × p', values: { L: totalLayers, W: sw, d_h: hd, p: precB }, resultValue: windowBytes, bar: [{ type: 'window', bytes: windowBytes }], ibarVal: fmtBytes(windowBytes) },
      { name: 'KV_cmp', tip: 'Source-owned packed KV. FP4 includes one E4M3 scale byte per 16 channels.', expr: 'Σ_sources ⌊T/r⌋ × slot_bytes', values: { T: tokens, slot_bytes: kvSlotBytes }, resultValue: compressedBytesV41, bar: [{ type: 'compressed', bytes: compressedBytesV41 }], ibarVal: fmtBytes(compressedBytesV41) },
      { name: 'KV', tip: 'Main CED cache: all-layer sliding window plus the four source-owned compressed caches.', expr: 'KV_sw + KV_cmp', values: { KV_sw: windowBytes, KV_cmp: compressedBytesV41 }, resultValue: mainKvBytes, bar: [{ type: 'window', bytes: windowBytes }, { type: 'compressed', bytes: compressedBytesV41 }], ibarVal: fmtBytes(mainKvBytes) },
      { name: 'Idx', tip: 'Four K owners, not eight querying layers. FP4 includes one E8M0 scale byte per 32 channels.', expr: 'Σ_K_owners ⌊T/r⌋ × slot_bytes', values: { T: tokens, slot_bytes: idxSlotBytes }, resultValue: idxBytes, bar: [{ type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(idxBytes) },
      { name: 'Total', tip: 'Combined cache payload for one sequence, including optional DSpark draft layers.', expr: 'KV + KV_draft + Idx', values: { KV: mainKvBytes, KV_draft: draftKvBytes, Idx: idxBytes }, resultValue: kvBytes + idxBytes, bar: [{ type: 'window', bytes: windowBytes + draftKvBytes }, { type: 'compressed', bytes: compressedBytesV41 }, { type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(seqs * (kvBytes + idxBytes)) }
    ];

    patterns = [
      { segs: [{ type: 'window', ratio: 1 }], count: totalLayers, label: 'window on every layer', bytes: sw * hd * precB },
    ];
    if (kvSourcesWithCache > 0) {
      patterns.push({ segs: [{ type: 'compressed', ratio: 1 }], count: kvSourcesWithCache, label: 'shared KV source cache', bytes: compressedBytesV41 / kvSourcesWithCache });
    }
    if (idxOwners.length > 0) {
      patterns.push({ segs: [{ type: 'indexer', ratio: 1 }], count: idxOwners.length, label: 'indexer K owner cache', bytes: idxBytes / idxOwners.length });
    }
    legendTypes = ['window', 'compressed', 'indexer'];

    breakdown = [
      { label: 'CED backbone layers', value: fmtNum(totalLayers) },
      { label: 'Sliding window', value: fmtNum(sw) },
      { label: 'KV source layers', value: fmtNum(kvSources.length), tip: 'Persistent compressed KV is allocated once per source layer and shared by its consumers.' },
      { label: 'Compressed source caches', value: fmtNum(kvSourcesWithCache) },
      { label: 'Sliding-window elements', value: fmtNum(windowElements) },
      { label: 'Compressed elements', value: fmtNum(compressedElementsV41), tip: 'Sum over the four source-owned caches, not over all consuming layers.' },
      { label: 'Indexer source layers', value: fmtNum(idxSources.length) },
      { label: 'Indexer K storage owners', value: fmtNum(idxOwners.length), tip: 'Other indexing layers reuse these keys rather than allocating another cache.' },
      { label: 'Global cache bytes/token', value: fmtNum((compressedBytesV41 + idxBytes) / tokens), tip: 'Excludes fixed sliding windows and Draft. Select FP4 for BOTH KV and Indexer to reproduce 890 bytes/token at even token counts; includes packed scale bytes.' },
      { label: 'Packed global KV slot bytes', value: fmtNum(kvSlotBytes) },
      { label: 'Packed indexer slot bytes', value: fmtNum(idxSlotBytes) },
      { label: 'Indexer elements', value: fmtNum(idxElementsV41), tip: 'One index key vector per compressed slot; index_n_heads is not a storage multiplier.' },
      { label: 'Draft layers included', value: fmtNum(draftLayersV41), tip: 'Optional DSpark/MTP layers add sliding-window state only.' },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'Indexer precision bytes', value: idxB.toString() },
      { label: 'Total bytes', value: fmtNum(kvBytes + idxBytes) },
    ];

  // ── mixed_full_sliding_gqa ──
  } else if (formula === 'mixed_full_sliding_gqa') {
    const fullLayers = f.full_attention_layers;
    const slidingLayers = f.sliding_attention_layers;
    const totalLayers = f.num_hidden_layers;
    const kvHeads = f.num_key_value_heads;
    const hd = f.head_dim;
    const globalHd = f.global_head_dim || hd;
    const globalKvHeads = f.num_global_key_value_heads || kvHeads;
    const sw = f.sliding_window;
    const storedLayers = f.stored_layers || totalLayers;

    const fullVHd = f.full_v_head_dim || f.v_head_dim || globalHd;
    const fullElements = fullLayers * globalKvHeads * (globalHd + fullVHd) * tokens;
    const fullBytes = fullElements * precB;

    const slidingVHd = f.sliding_v_head_dim || f.swa_v_head_dim || hd;
    const slidingKvHeads = f.sliding_num_key_value_heads || f.swa_num_key_value_heads || kvHeads;
    const slidingHd = f.sliding_head_dim || f.swa_head_dim || hd;
    const retainedTokens = Math.min(tokens, sw);
    const slidingElements = slidingLayers * slidingKvHeads * (slidingHd + slidingVHd) * retainedTokens;

    const kvElements = fullElements + slidingElements;
    kvBytes = kvElements * precB;
    perTokenBytes = (fullLayers * globalKvHeads * (globalHd + fullVHd) + slidingLayers * slidingKvHeads * (slidingHd + slidingVHd) * (retainedTokens / tokens)) * precB;

    formulaTitle = model.label + ' mixed full + sliding window attention';
    var mfsSlidingBytes = slidingElements * precB;
    formulas = [
      { name: 'KV_f', tip: 'Full attention layers store KV for the entire context length.', expr: 'L_f \u00d7 h_f \u00d7 (d_f + d_vf) \u00d7 T \u00d7 p', values: { L_f: fullLayers, h_f: globalKvHeads, d_f: globalHd, d_vf: fullVHd, T: tokens, p: precB }, resultValue: fullBytes, bar: [{ type: 'full', bytes: fullBytes }], ibarVal: fmtBytes(fullBytes) },
      { name: 'KV_s', tip: 'Sliding window attention layers only store KV for the local window.', expr: 'L_s \u00d7 h_s \u00d7 (d_s + d_vs) \u00d7 min(T, W) \u00d7 p', values: { L_s: slidingLayers, h_s: slidingKvHeads, d_s: slidingHd, d_vs: slidingVHd, T: tokens, W: sw, p: precB }, resultValue: mfsSlidingBytes, bar: [{ type: 'window', bytes: mfsSlidingBytes }], ibarVal: fmtBytes(mfsSlidingBytes) },
      { name: 'KV', tip: 'Combined KV cache for both attention types.', expr: 'KV_f + KV_s', values: { KV_f: fullBytes, KV_s: mfsSlidingBytes }, resultValue: kvBytes, bar: [{ type: 'full', bytes: fullBytes }, { type: 'window', bytes: mfsSlidingBytes }], ibarVal: fmtBytes(kvBytes) },
      { name: 'Total', tip: 'Combined cache payload for all concurrent sequences.', expr: 'B \u00d7 KV', values: { KV: kvBytes }, resultValue: kvBytes, bar: [{ type: 'full', bytes: fullBytes }, { type: 'window', bytes: mfsSlidingBytes }], ibarVal: fmtBytes(seqs * kvBytes) }
    ];
    var mfsFullDenom = globalHd + fullVHd;
    var mfsSlidingDenom = slidingHd + slidingVHd;
    patterns = [
      {
        segs: [{ type: 'full', ratio: globalHd / mfsFullDenom }, { type: 'full-alt', ratio: fullVHd / mfsFullDenom }],
        count: fullLayers,
        label: 'full attn',
        bytes: fullBytes / fullLayers
      },
      {
        segs: [{ type: 'window', ratio: slidingHd / mfsSlidingDenom }, { type: 'window-alt', ratio: slidingVHd / mfsSlidingDenom }],
        count: slidingLayers,
        label: 'sliding attn',
        bytes: mfsSlidingBytes / slidingLayers
      }
    ];
    legendTypes = ['full', 'window'];

    breakdown = [
      { label: 'Full attention layers', value: fmtNum(fullLayers) },
      { label: 'Sliding attention layers', value: fmtNum(slidingLayers) },
      { label: 'Global KV heads', value: fmtNum(globalKvHeads) },
      { label: 'Global K head dim', value: fmtNum(globalHd) },
      { label: 'Global V head dim', value: fmtNum(fullVHd) },
      { label: 'Sliding KV heads', value: fmtNum(slidingKvHeads) },
      { label: 'Sliding K head dim', value: fmtNum(slidingHd) },
      { label: 'Sliding V head dim', value: fmtNum(slidingVHd) },
      { label: 'Sliding window', value: fmtNum(sw) },
      { label: 'Retained sliding tokens', value: fmtNum(retainedTokens), tip: 'min(tokens, sliding_window)' },
      { label: 'Full KV elements', value: fmtNum(fullElements) },
      { label: 'Sliding KV elements', value: fmtNum(slidingElements) },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'Total bytes', value: fmtNum(kvBytes) },
    ];

  // ── qwen_linear_full_hybrid ──
  } else if (formula === 'qwen_linear_full_hybrid') {
    const fullLayers = f.full_attention_layers;
    const linearLayers = f.linear_attention_layers;
    const totalLayers = f.num_hidden_layers;
    const kvHeads = f.num_key_value_heads;
    const hd = f.head_dim;
    const linKvHeads = f.linear_num_key_heads;
    const linValHeads = f.linear_num_value_heads;
    const linKeyHd = f.linear_key_head_dim;
    const linValHd = f.linear_value_head_dim;
    const convDim = f.linear_conv_kernel_dim;

    // Full attention (standard GQA)
    const fullElements = 2 * fullLayers * kvHeads * hd * tokens;
    const fullBytes = fullElements * precB;

    // Linear attention conv state: fixed per sequence, BF16 (2 bytes)
    const linConvElements = linearLayers * convDim * (2 * linKvHeads * linKeyHd + linValHeads * linValHd);
    const linConvBytes = includeLinear ? linConvElements * 2 : 0;

    // Linear attention recurrent state: fixed per sequence, FP32 (4 bytes)
    const linRecurrentElements = linearLayers * linValHeads * linKeyHd * linValHd;
    const linRecurrentBytes = includeLinear ? linRecurrentElements * 4 : 0;

    const linBytes = linConvBytes + linRecurrentBytes;
    kvBytes = fullBytes + linBytes;

    perTokenBytes = (fullBytes + linBytes) / tokens;

    formulaTitle = model.label + ' linear + full attention hybrid';
    formulas = [
      { name: 'KV_f', tip: 'Full attention layers use standard GQA KV cache for the entire context.', expr: '2 \u00d7 L_f \u00d7 h_kv \u00d7 d_h \u00d7 T \u00d7 p', values: { L_f: fullLayers, h_kv: kvHeads, d_h: hd, T: tokens, p: precB }, resultValue: fullBytes, bar: [{ type: 'full', bytes: fullBytes }], ibarVal: fmtBytes(fullBytes) },
      { name: 'S_conv', tip: 'Linear attention conv kernel state, fixed per sequence in BF16.', expr: 'B \u00d7 L_l \u00d7 k_c \u00d7 (2 \u00d7 h_kl \u00d7 d_kl + h_vl \u00d7 d_vl) \u00d7 2', values: { L_l: linearLayers, k_c: convDim, h_kl: linKvHeads, d_kl: linKeyHd, h_vl: linValHeads, d_vl: linValHd }, resultValue: linConvBytes, bar: [{ type: 'fixed', bytes: linConvBytes }], ibarVal: fmtBytes(linConvBytes) },
      { name: 'S_rec', tip: 'Linear attention recurrent state, fixed per sequence in FP32.', expr: 'B \u00d7 L_l \u00d7 h_vl \u00d7 d_kl \u00d7 d_vl \u00d7 4', values: { L_l: linearLayers, h_vl: linValHeads, d_kl: linKeyHd, d_vl: linValHd }, resultValue: linRecurrentBytes, bar: [{ type: 'fixed-alt', bytes: linRecurrentBytes }], ibarVal: fmtBytes(linRecurrentBytes) },
      { name: 'Total', tip: 'Combined full + linear KV cache.', expr: 'KV_f + S_conv + S_rec', values: { KV_f: fullBytes, S_conv: linConvBytes, S_rec: linRecurrentBytes }, resultValue: kvBytes, bar: [{ type: 'full', bytes: fullBytes }, { type: 'fixed', bytes: linConvBytes }, { type: 'fixed-alt', bytes: linRecurrentBytes }], ibarVal: fmtBytes(seqs * kvBytes) }
    ];
    var qlfLinearDenom = linConvBytes + linRecurrentBytes;
    patterns = [
      {
        segs: [{ type: 'full', ratio: 0.5 }, { type: 'full-alt', ratio: 0.5 }],
        count: fullLayers,
        label: 'full attn',
        bytes: fullBytes / fullLayers
      },
      {
        segs: qlfLinearDenom > 0
          ? [{ type: 'fixed', ratio: linConvBytes / qlfLinearDenom }, { type: 'fixed-alt', ratio: linRecurrentBytes / qlfLinearDenom }]
          : [{ type: 'fixed', ratio: 0.5 }, { type: 'fixed-alt', ratio: 0.5 }],
        count: linearLayers,
        label: 'linear attn',
        bytes: qlfLinearDenom > 0 ? linBytes / linearLayers : 0
      }
    ];
    legendTypes = ['full', 'fixed'];

    breakdown = [
      { label: 'Full attention layers', value: fmtNum(fullLayers) },
      { label: 'Linear attention layers', value: fmtNum(linearLayers) },
      { label: 'Linear state included', value: includeLinear ? 'Yes' : 'No', tip: 'Whether linear attention conv + recurrent state is included in the calculation.' },
      { label: 'Full KV heads', value: fmtNum(kvHeads) },
      { label: 'Full head dim', value: fmtNum(hd) },
      { label: 'Full KV elements', value: fmtNum(fullElements) },
      { label: 'Linear conv elements', value: fmtNum(linConvElements) },
      { label: 'Linear recurrent elements', value: fmtNum(linRecurrentElements) },
      { label: 'Per-token elements', value: fmtNum(2 * fullLayers * kvHeads * hd), tip: 'Full attention only: 2 \u00d7 full_layers \u00d7 kv_heads \u00d7 head_dim' },
      { label: 'Precision bytes', value: precB.toString() },
      { label: 'Total bytes', value: fmtNum(kvBytes) },
    ];

  // ── glm5_next_hybrid (GLM-5.3-Flash: KDA linear + sparse MLA) ──
  } else if (formula === 'glm5_next_hybrid') {
    const totalLayers = f.num_hidden_layers;
    const layerTypes = Array.isArray(f.layer_types) ? f.layer_types : [];
    const configuredSparseLayerIds = Array.isArray(f.sparse_attention_layer_ids) ? f.sparse_attention_layer_ids : [];
    const sparseLayerIds = configuredSparseLayerIds.length > 0
      ? configuredSparseLayerIds
      : (layerTypes.length > 0
        ? layerTypes.map(function (type, index) { return type === 'deepseek_sparse_attention' ? index : -1; }).filter(function (index) { return index >= 0; })
        : []);
    const sparseLayers = sparseLayerIds.length || f.sparse_attention_layers || 0;
    const linearLayers = f.linear_attention_layers || (totalLayers - sparseLayers);
    const sparseKvRank = f.kv_lora_rank || 0;
    const qkRopeHd = f.qk_rope_head_dim || 0;
    const linHeads = f.linear_num_heads || 0;
    const linHd = f.linear_head_dim || 0;
    const convKernel = f.linear_conv_kernel_dim || 0;

    // Sparse MLA layers retain one compressed latent per token. GLM-5.3-Flash
    // has qk_rope_head_dim=0, so there is no additional RoPE cache segment.
    const sparseKvElements = sparseLayers * (sparseKvRank + qkRopeHd) * tokens;
    const sparseKvBytes = sparseKvElements * precB;

    // KDA state is fixed per sequence. The GLM implementation projects Q, K
    // and V with the same (heads × head_dim) width, then keeps a recurrent
    // [heads, key_dim, value_dim] state. These states are not token-linear KV.
    const linearQkvDim = linHeads * linHd;
    const linearConvElements = linearLayers * convKernel * (3 * linearQkvDim);
    const linearRecurrentElements = linearLayers * linHeads * linHd * linHd;
    const linearConvBytes = includeLinear ? linearConvElements * 2 : 0;
    const linearRecurrentBytes = includeLinear ? linearRecurrentElements * 4 : 0;
    const linearStateBytes = linearConvBytes + linearRecurrentBytes;

    kvLayerBytes = [];
    var linearStatePerLayer = linearLayers > 0 ? linearStateBytes / linearLayers : 0;
    for (var glmLayer = 0; glmLayer < totalLayers; glmLayer++) {
      var isSparseGlmLayer = sparseLayerIds.length > 0
        ? sparseLayerIds.indexOf(glmLayer) >= 0
        : glmLayer < sparseLayers;
      kvLayerBytes[glmLayer] = isSparseGlmLayer
        ? (sparseKvRank + qkRopeHd) * tokens * precB
        : linearStatePerLayer;
    }
    mainKvBytes = sparseKvBytes + linearStateBytes;
    linearLayerBytes = kvLayerBytes.map(function (bytes, layer) {
      var sparse = sparseLayerIds.length > 0 ? sparseLayerIds.indexOf(layer) >= 0 : layer < sparseLayers;
      return sparse ? 0 : bytes;
    });

    // The optional draft stack is represented as a latent MLA layer. The
    // published Transformers block has no separate indexer for that stack.
    var draftLayersGlm = includeDraft ? (f.num_nextn_predict_layers || 0) : 0;
    draftKvBytes = draftLayersGlm * (sparseKvRank + qkRopeHd) * tokens * precB;
    kvBytes = mainKvBytes + draftKvBytes;

    // DSA indexer keys are pooled once per index_kpool tokens. index_topk is a
    // lookup budget, not the number of stored cache vectors. The physical
    // tail uses two BF16 vectors (K plus gate score) for the incomplete pool.
    const indexPool = f.index_kpool || 1;
    const indexHeadDim = f.index_head_dim || 0;
    const fullIndexPools = Math.floor(tokens / indexPool);
    const indexTailTokens = tokens % indexPool;
    const indexPoolElements = sparseLayers * fullIndexPools * indexHeadDim;
    const indexTailElements = sparseLayers * indexTailTokens * 2 * indexHeadDim;
    const indexScaleBytesPerPool = idxB === 1 ? (f.indexer_scale_bytes || 0) : 0;
    const indexScaleBytes = sparseLayers * fullIndexPools * indexScaleBytesPerPool;
    const indexPoolBytes = (indexPoolElements * idxB) + indexScaleBytes;
    const indexTailBytes = indexTailElements * 2;
    idxBytes = indexPoolBytes + indexTailBytes;
    idxLayers = sparseLayers;
    idxLayerBytes = [];
    sparseLayerIds.forEach(function (layerId) {
      idxLayerBytes[layerId] = fullIndexPools * (indexHeadDim * idxB + indexScaleBytesPerPool) + indexTailTokens * 2 * indexHeadDim * 2;
    });
    if (sparseLayerIds.length === 0) {
      for (var glmIdxLayer = 0; glmIdxLayer < sparseLayers; glmIdxLayer++) {
        idxLayerBytes[glmIdxLayer] = fullIndexPools * (indexHeadDim * idxB + indexScaleBytesPerPool) + indexTailTokens * 2 * indexHeadDim * 2;
      }
    }
    perTokenBytes = (kvBytes + idxBytes) / tokens;

    formulaTitle = model.label + ' KDA linear + sparse MLA';
    const sparsePerLayerBytes = (sparseKvRank + qkRopeHd) * tokens * precB;
    const indexPerSparseLayerBytes = sparseLayers > 0 ? idxBytes / sparseLayers : 0;
    const linearStateDenom = linearConvBytes + linearRecurrentBytes;
    formulas = [
      { name: 'KV_s', tip: 'Sparse MLA layers retain one compressed latent per token. qk_rope_head_dim=0 for GLM-5.3-Flash.', expr: 'L_s × (d_c + d_r) × T × p', values: { L_s: sparseLayers, d_c: sparseKvRank, d_r: qkRopeHd, T: tokens, p: precB }, resultValue: sparseKvBytes, bar: [{ type: 'compressed', bytes: sparseKvBytes }], ibarVal: fmtBytes(sparseKvBytes) },
      { name: 'S_conv', tip: 'KDA short-convolution state, fixed per sequence in BF16.', expr: 'B × L_l × k_c × (3 × h_l × d_l) × 2', values: { L_l: linearLayers, k_c: convKernel, h_l: linHeads, d_l: linHd }, resultValue: linearConvBytes, bar: [{ type: 'fixed', bytes: linearConvBytes }], ibarVal: fmtBytes(linearConvBytes) },
      { name: 'S_rec', tip: 'KDA recurrent state, fixed per sequence in FP32.', expr: 'B × L_l × h_l × d_l × d_l × 4', values: { L_l: linearLayers, h_l: linHeads, d_l: linHd }, resultValue: linearRecurrentBytes, bar: [{ type: 'fixed-alt', bytes: linearRecurrentBytes }], ibarVal: fmtBytes(linearRecurrentBytes) },
      { name: 'Idx', tip: 'Sparse indexer cache: one K vector per complete token pool plus a small BF16 tail for the incomplete pool. The FP8 path also stores one scale scalar per pool; index_topk is not a cache-length multiplier.', expr: 'L_s × (⌊T/k_p⌋ × (d_idx × p_idx + s_idx) + (T mod k_p) × 2 × d_idx × 2)', values: { L_s: sparseLayers, T: tokens, k_p: indexPool, d_idx: indexHeadDim, p_idx: idxB, s_idx: indexScaleBytesPerPool }, resultValue: idxBytes, bar: [{ type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(idxBytes) },
      { name: 'KV', tip: 'Main cache combining sparse MLA token cache and optional KDA fixed state.', expr: 'KV_s + S_conv + S_rec', values: { KV_s: sparseKvBytes, S_conv: linearConvBytes, S_rec: linearRecurrentBytes }, resultValue: mainKvBytes, bar: [{ type: 'compressed', bytes: sparseKvBytes }, { type: 'fixed', bytes: linearConvBytes }, { type: 'fixed-alt', bytes: linearRecurrentBytes }], ibarVal: fmtBytes(mainKvBytes) },
      { name: 'Total', tip: 'Combined cache payload for one sequence, including optional draft layers.', expr: 'KV + KV_draft + Idx', values: { KV: mainKvBytes, KV_draft: draftKvBytes, Idx: idxBytes }, resultValue: kvBytes + idxBytes, bar: [{ type: 'compressed', bytes: sparseKvBytes + draftKvBytes }, { type: 'fixed', bytes: linearConvBytes }, { type: 'fixed-alt', bytes: linearRecurrentBytes }, { type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(seqs * (kvBytes + idxBytes)) }
    ];

    patterns = [];
    if (sparseLayers > 0) {
      var sparsePatternTotal = sparsePerLayerBytes + indexPerSparseLayerBytes;
      patterns.push({ segs: sparsePatternTotal > 0 ? [{ type: 'compressed', ratio: sparsePerLayerBytes / sparsePatternTotal }, { type: 'indexer', ratio: indexPerSparseLayerBytes / sparsePatternTotal }] : [{ type: 'compressed', ratio: 1 }], count: sparseLayers, label: 'sparse MLA + indexer', bytes: sparsePatternTotal });
    }
    if (linearLayers > 0) {
      patterns.push({ segs: linearStateDenom > 0 ? [{ type: 'fixed', ratio: linearConvBytes / linearStateDenom }, { type: 'fixed-alt', ratio: linearRecurrentBytes / linearStateDenom }] : [{ type: 'fixed', ratio: 0.5 }, { type: 'fixed-alt', ratio: 0.5 }], count: linearLayers, label: 'KDA linear state', bytes: linearStateDenom > 0 ? linearStateDenom / linearLayers : 0 });
    }
    legendTypes = ['compressed', 'indexer', 'fixed'];

    breakdown = [
      { label: 'Layers', value: fmtNum(totalLayers) },
      { label: 'Sparse MLA layers', value: fmtNum(sparseLayers) },
      { label: 'KDA linear layers', value: fmtNum(linearLayers) },
      { label: 'KV LoRA rank', value: fmtNum(sparseKvRank) },
      { label: 'QK RoPE head dim', value: fmtNum(qkRopeHd) },
      { label: 'Sparse KV elements', value: fmtNum(sparseKvElements) },
      { label: 'Linear state included', value: includeLinear ? 'Yes' : 'No', tip: 'Whether the fixed KDA convolution and recurrent state is included.' },
      { label: 'Linear conv elements', value: fmtNum(linearConvElements) },
      { label: 'Linear recurrent elements', value: fmtNum(linearRecurrentElements) },
      { label: 'Indexer pool size', value: fmtNum(indexPool), tip: 'One indexer K vector is stored per complete pool; index_topk=2048 is a lookup budget.' },
      { label: 'Indexer pooled elements', value: fmtNum(indexPoolElements) },
      { label: 'Indexer scale bytes per pool', value: fmtNum(indexScaleBytesPerPool), tip: 'GLM FP8 indexer cache stores one scale scalar per pooled K vector.' },
      { label: 'Indexer tail elements', value: fmtNum(indexTailElements), tip: 'Small incomplete-pool tail, stored in BF16 K plus gate-score state.' },
      { label: 'Indexer precision bytes', value: idxB.toString() },
      { label: 'Draft layers included', value: fmtNum(draftLayersGlm), tip: 'Optional latent draft layers; no separate draft indexer cache is added.' },
      { label: 'Total bytes', value: fmtNum(kvBytes + idxBytes) },
    ];

  // ── qwen_qsa_gdn_hybrid (Qwen3.8-Flash-Next) ──
  } else if (formula === 'qwen_qsa_gdn_hybrid') {
    const totalLayers = f.num_hidden_layers;
    const fullLayerIds = Array.isArray(f.full_attention_layer_ids)
      ? f.full_attention_layer_ids
      : [];
    const fullLayers = fullLayerIds.length || f.full_attention_layers || 0;
    const linearLayers = f.linear_attention_layers || (totalLayers - fullLayers);
    const kvHeads = f.num_key_value_heads;
    const hd = f.head_dim;
    const linKeyHeads = f.linear_num_key_heads;
    const linValueHeads = f.linear_num_value_heads;
    const linKeyHd = f.linear_key_head_dim;
    const linValueHd = f.linear_value_head_dim;
    const convKernel = f.linear_conv_kernel_dim;

    // QSA full-attention layers use raw GQA K and V. This is not MLA, so KV
    // is sharded by TP in deployment and has the usual two (K + V) factor.
    const fullElementsQsa = 2 * fullLayers * kvHeads * hd * tokens;
    const fullBytesQsa = fullElementsQsa * precB;

    // GDN linear state is fixed per sequence and uses BF16 conv + FP32
    // recurrent state, matching the Qwen linear-attention implementation.
    const linearConvElementsQsa = linearLayers * convKernel * (2 * linKeyHeads * linKeyHd + linValueHeads * linValueHd);
    const linearRecurrentElementsQsa = linearLayers * linValueHeads * linKeyHd * linValueHd;
    const linearConvBytesQsa = includeLinear ? linearConvElementsQsa * 2 : 0;
    const linearRecurrentBytesQsa = includeLinear ? linearRecurrentElementsQsa * 4 : 0;
    const linearStateBytesQsa = linearConvBytesQsa + linearRecurrentBytesQsa;

    kvLayerBytes = [];
    const fullIdSetQsa = {};
    fullLayerIds.forEach(function (layerId) { fullIdSetQsa[layerId] = true; });
    var linearStatePerLayerQsa = linearLayers > 0 ? linearStateBytesQsa / linearLayers : 0;
    for (var qsaLayer = 0; qsaLayer < totalLayers; qsaLayer++) {
      var isFullQsaLayer = fullLayerIds.length > 0 ? !!fullIdSetQsa[qsaLayer] : qsaLayer >= 0 && ((qsaLayer + 1) % (f.full_attention_interval || 4) === 0);
      kvLayerBytes[qsaLayer] = isFullQsaLayer
        ? 2 * kvHeads * hd * tokens * precB
        : linearStatePerLayerQsa;
    }
    mainKvBytes = fullBytesQsa + linearStateBytesQsa;
    linearLayerBytes = kvLayerBytes.map(function (bytes, layer) {
      var full = fullLayerIds.length > 0 ? !!fullIdSetQsa[layer] : (layer + 1) % (f.full_attention_interval || 4) === 0;
      return full ? 0 : bytes;
    });

    var draftLayersQsa = includeDraft ? (f.mtp_num_hidden_layers || 0) : 0;
    var draftFullLayersQsa = includeDraft ? (f.mtp_full_attention_layers || draftLayersQsa) : 0;
    draftKvBytes = draftFullLayersQsa * 2 * kvHeads * hd * tokens * precB;
    kvBytes = mainKvBytes + draftKvBytes;

    // QSA's indexer stores one K vector per compressed micro-block. The four
    // indexer heads are query-side heads; indexer_kv_heads=1 is the cache
    // multiplier. Keep the BF16 tail separate from the selectable pool
    // precision because serving engines retain it in BF16.
    const indexRatioQsa = f.indexer_compress_ratio || 1;
    const indexHdQsa = f.indexer_head_dim || 0;
    const indexKvHeadsQsa = f.indexer_kv_heads || 1;
    const fullPoolsQsa = Math.floor(tokens / indexRatioQsa);
    const tailTokensQsa = tokens % indexRatioQsa;
    const indexPoolElementsQsa = fullLayers * fullPoolsQsa * indexKvHeadsQsa * indexHdQsa;
    const indexTailElementsQsa = fullLayers * tailTokensQsa * 2 * indexHdQsa;
    const indexPoolBytesQsa = indexPoolElementsQsa * idxB;
    const indexTailBytesQsa = indexTailElementsQsa * 2;
    idxBytes = indexPoolBytesQsa + indexTailBytesQsa;
    idxLayers = fullLayers;
    idxLayerBytes = [];
    if (fullLayerIds.length > 0) {
      fullLayerIds.forEach(function (layerId) {
        idxLayerBytes[layerId] = fullPoolsQsa * indexKvHeadsQsa * indexHdQsa * idxB + tailTokensQsa * 2 * indexHdQsa * 2;
      });
    } else {
      var generatedFullIdsQsa = [];
      for (var qsaFullIndex = (f.full_attention_interval || 4) - 1; qsaFullIndex < totalLayers; qsaFullIndex += (f.full_attention_interval || 4)) generatedFullIdsQsa.push(qsaFullIndex);
      generatedFullIdsQsa.slice(0, fullLayers).forEach(function (layerId) {
        idxLayerBytes[layerId] = fullPoolsQsa * indexKvHeadsQsa * indexHdQsa * idxB + tailTokensQsa * 2 * indexHdQsa * 2;
      });
    }
    perTokenBytes = (kvBytes + idxBytes) / tokens;

    formulaTitle = model.label + ' GDN linear + QSA';
    const fullPerLayerQsa = 2 * kvHeads * hd * tokens * precB;
    const indexPerFullLayerQsa = fullLayers > 0 ? idxBytes / fullLayers : 0;
    const linearStateDenomQsa = linearConvBytesQsa + linearRecurrentBytesQsa;
    formulas = [
      { name: 'KV_f', tip: 'QSA full-attention layers store raw GQA K and V for the full context.', expr: '2 × L_f × h_kv × d_h × T × p', values: { L_f: fullLayers, h_kv: kvHeads, d_h: hd, T: tokens, p: precB }, resultValue: fullBytesQsa, bar: [{ type: 'full', bytes: fullBytesQsa }], ibarVal: fmtBytes(fullBytesQsa) },
      { name: 'S_conv', tip: 'GDN short-convolution state, fixed per sequence in BF16.', expr: 'B × L_l × k_c × (2 × h_kl × d_kl + h_vl × d_vl) × 2', values: { L_l: linearLayers, k_c: convKernel, h_kl: linKeyHeads, d_kl: linKeyHd, h_vl: linValueHeads, d_vl: linValueHd }, resultValue: linearConvBytesQsa, bar: [{ type: 'fixed', bytes: linearConvBytesQsa }], ibarVal: fmtBytes(linearConvBytesQsa) },
      { name: 'S_rec', tip: 'GDN recurrent state, fixed per sequence in FP32.', expr: 'B × L_l × h_vl × d_kl × d_vl × 4', values: { L_l: linearLayers, h_vl: linValueHeads, d_kl: linKeyHd, d_vl: linValueHd }, resultValue: linearRecurrentBytesQsa, bar: [{ type: 'fixed-alt', bytes: linearRecurrentBytesQsa }], ibarVal: fmtBytes(linearRecurrentBytesQsa) },
      { name: 'Idx', tip: 'QSA indexer cache: one K vector per compressed micro-block, plus a small BF16 tail. indexer_n_heads and indexer_budget are not cache multipliers.', expr: 'L_f × (⌊T/r_idx⌋ × h_idx_kv × d_idx × p_idx + (T mod r_idx) × 2 × d_idx × 2)', values: { L_f: fullLayers, T: tokens, r_idx: indexRatioQsa, h_idx_kv: indexKvHeadsQsa, d_idx: indexHdQsa, p_idx: idxB }, resultValue: idxBytes, bar: [{ type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(idxBytes) },
      { name: 'KV', tip: 'Main QSA + GDN cache for one sequence.', expr: 'KV_f + S_conv + S_rec', values: { KV_f: fullBytesQsa, S_conv: linearConvBytesQsa, S_rec: linearRecurrentBytesQsa }, resultValue: mainKvBytes, bar: [{ type: 'full', bytes: fullBytesQsa }, { type: 'fixed', bytes: linearConvBytesQsa }, { type: 'fixed-alt', bytes: linearRecurrentBytesQsa }], ibarVal: fmtBytes(mainKvBytes) },
      { name: 'Total', tip: 'Combined cache payload for one sequence, including optional MTP full-attention cache.', expr: 'KV + KV_draft + Idx', values: { KV: mainKvBytes, KV_draft: draftKvBytes, Idx: idxBytes }, resultValue: kvBytes + idxBytes, bar: [{ type: 'full', bytes: fullBytesQsa + draftKvBytes }, { type: 'fixed', bytes: linearConvBytesQsa }, { type: 'fixed-alt', bytes: linearRecurrentBytesQsa }, { type: 'indexer', bytes: idxBytes }], ibarVal: fmtBytes(seqs * (kvBytes + idxBytes)) }
    ];

    patterns = [];
    if (fullLayers > 0) {
      var qsaFullPatternTotal = fullPerLayerQsa + indexPerFullLayerQsa;
      patterns.push({ segs: qsaFullPatternTotal > 0 ? [{ type: 'full', ratio: fullPerLayerQsa / qsaFullPatternTotal }, { type: 'indexer', ratio: indexPerFullLayerQsa / qsaFullPatternTotal }] : [{ type: 'full', ratio: 1 }], count: fullLayers, label: 'QSA full + indexer', bytes: qsaFullPatternTotal });
    }
    if (linearLayers > 0) {
      patterns.push({ segs: linearStateDenomQsa > 0 ? [{ type: 'fixed', ratio: linearConvBytesQsa / linearStateDenomQsa }, { type: 'fixed-alt', ratio: linearRecurrentBytesQsa / linearStateDenomQsa }] : [{ type: 'fixed', ratio: 0.5 }, { type: 'fixed-alt', ratio: 0.5 }], count: linearLayers, label: 'GDN linear state', bytes: linearStateDenomQsa > 0 ? linearStateDenomQsa / linearLayers : 0 });
    }
    legendTypes = ['full', 'indexer', 'fixed'];

    var ngramTableElementsQsa = (f.ngram_vocab_size_base || 0) * (f.ple_embed_dim || 0);
    breakdown = [
      { label: 'Layers', value: fmtNum(totalLayers) },
      { label: 'QSA full layers', value: fmtNum(fullLayers) },
      { label: 'GDN linear layers', value: fmtNum(linearLayers) },
      { label: 'Full KV heads', value: fmtNum(kvHeads) },
      { label: 'Full head dim', value: fmtNum(hd) },
      { label: 'Full KV elements', value: fmtNum(fullElementsQsa) },
      { label: 'Linear state included', value: includeLinear ? 'Yes' : 'No', tip: 'Whether the fixed GDN convolution and recurrent state is included.' },
      { label: 'Linear conv elements', value: fmtNum(linearConvElementsQsa) },
      { label: 'Linear recurrent elements', value: fmtNum(linearRecurrentElementsQsa) },
      { label: 'Indexer compress ratio', value: fmtNum(indexRatioQsa), tip: 'One indexer K vector per compressed micro-block.' },
      { label: 'Indexer pooled elements', value: fmtNum(indexPoolElementsQsa) },
      { label: 'Indexer tail elements', value: fmtNum(indexTailElementsQsa), tip: 'Small incomplete-pool tail, retained in BF16.' },
      { label: 'Indexer precision bytes', value: idxB.toString() },
      { label: 'Draft full-attention layers', value: fmtNum(draftFullLayersQsa), tip: 'Optional MTP layer(s); the nested Qwen MTP config is full attention and has no separate indexer cache.' },
      { label: 'N-gram / PLE table elements', value: fmtNum(ngramTableElementsQsa), tip: 'Model auxiliary table; off-accelerator and excluded from per-sequence KV cache.' },
      { label: 'Total bytes', value: fmtNum(kvBytes + idxBytes) },
    ];


  // ── kda_gated_mla (Kimi K3: KDA linear attention + Gated MLA) ──
  } else if (formula === 'kda_gated_mla') {
    const fullLayers = f.full_attention_layers;
    const linearLayers = f.linear_attention_layers;
    const kvLoraRank = f.kv_lora_rank;
    const qkRopeHd = f.qk_rope_head_dim || 0;
    const linKvHeads = f.linear_num_key_heads;
    const linValHeads = f.linear_num_value_heads;
    const linKeyHd = f.linear_key_head_dim;
    const linValHd = f.linear_value_head_dim;
    const convDim = f.linear_conv_kernel_dim;

    // Gated MLA full-attention layers: compressed latent KV per token
    const mlaElements = fullLayers * (kvLoraRank + qkRopeHd) * tokens;
    const mlaBytes = mlaElements * precB;

    // KDA linear-attention conv state: fixed per sequence, BF16 (2 bytes)
    const kdaConvElements = linearLayers * convDim * (2 * linKvHeads * linKeyHd + linValHeads * linValHd);
    const kdaConvBytes = includeLinear ? kdaConvElements * 2 : 0;

    // KDA delta-rule recurrent state: fixed per sequence, FP32 (4 bytes)
    const kdaRecurrentElements = linearLayers * linValHeads * linKeyHd * linValHd;
    const kdaRecurrentBytes = includeLinear ? kdaRecurrentElements * 4 : 0;

    const kdaBytes = kdaConvBytes + kdaRecurrentBytes;
    kvBytes = mlaBytes + kdaBytes;

    perTokenBytes = (mlaBytes + kdaBytes) / tokens;

    formulaTitle = model.label + ' KDA + Gated MLA attention';
    var kdaMlaCompressedBytes = mlaBytes * kvLoraRank / (kvLoraRank + qkRopeHd);
    var kdaMlaRopeBytes = mlaBytes * qkRopeHd / (kvLoraRank + qkRopeHd);
    var kdaMlaBar = qkRopeHd > 0
      ? [{ type: 'compressed', bytes: kdaMlaCompressedBytes }, { type: 'rope', bytes: kdaMlaRopeBytes }]
      : [{ type: 'compressed', bytes: mlaBytes }];
    formulas = [
      { name: 'KV_f', tip: 'Gated MLA full-attention layers store a compressed latent KV cache (kv_lora_rank + qk_rope_head_dim) per token.', expr: 'L_f \u00d7 (d_c + d_r) \u00d7 T \u00d7 p', values: { L_f: fullLayers, d_c: kvLoraRank, d_r: qkRopeHd, T: tokens, p: precB }, resultValue: mlaBytes, bar: kdaMlaBar, ibarVal: fmtBytes(mlaBytes) },
      { name: 'S_conv', tip: 'KDA short-convolution state, fixed per sequence in BF16. Include via the linear-state toggle.', expr: 'B \u00d7 L_l \u00d7 k_c \u00d7 (2 \u00d7 h_kl \u00d7 d_kl + h_vl \u00d7 d_vl) \u00d7 2', values: { L_l: linearLayers, k_c: convDim, h_kl: linKvHeads, d_kl: linKeyHd, h_vl: linValHeads, d_vl: linValHd }, resultValue: kdaConvBytes, bar: [{ type: 'fixed', bytes: kdaConvBytes }], ibarVal: fmtBytes(kdaConvBytes) },
      { name: 'S_rec', tip: 'KDA delta-rule recurrent state, fixed per sequence in FP32. Include via the linear-state toggle.', expr: 'B \u00d7 L_l \u00d7 h_vl \u00d7 d_kl \u00d7 d_vl \u00d7 4', values: { L_l: linearLayers, h_vl: linValHeads, d_kl: linKeyHd, d_vl: linValHd }, resultValue: kdaRecurrentBytes, bar: [{ type: 'fixed-alt', bytes: kdaRecurrentBytes }], ibarVal: fmtBytes(kdaRecurrentBytes) },
      { name: 'Total', tip: 'Combined Gated MLA + KDA cache (one sequence).', expr: 'KV_f + S_conv + S_rec', values: { KV_f: mlaBytes, S_conv: kdaConvBytes, S_rec: kdaRecurrentBytes }, resultValue: kvBytes, bar: [{ type: 'compressed', bytes: kdaMlaCompressedBytes }, { type: 'rope', bytes: kdaMlaRopeBytes }, { type: 'fixed', bytes: kdaConvBytes }, { type: 'fixed-alt', bytes: kdaRecurrentBytes }], ibarVal: fmtBytes(seqs * kvBytes) }
    ];
    var kdaLinearDenom = kdaConvBytes + kdaRecurrentBytes;
    patterns = [
      {
        segs: qkRopeHd > 0
          ? [{ type: 'compressed', ratio: kvLoraRank / (kvLoraRank + qkRopeHd) }, { type: 'rope', ratio: qkRopeHd / (kvLoraRank + qkRopeHd) }]
          : [{ type: 'compressed', ratio: 1 }],
        count: fullLayers,
        label: 'gated MLA',
        bytes: mlaBytes / fullLayers
      },
      {
        segs: kdaLinearDenom > 0
          ? [{ type: 'fixed', ratio: kdaConvBytes / kdaLinearDenom }, { type: 'fixed-alt', ratio: kdaRecurrentBytes / kdaLinearDenom }]
          : [{ type: 'fixed', ratio: 0.5 }, { type: 'fixed-alt', ratio: 0.5 }],
        count: linearLayers,
        label: 'KDA linear',
        bytes: kdaLinearDenom > 0 ? kdaBytes / linearLayers : 0
      }
    ];
    legendTypes = ['compressed', 'rope', 'fixed'];

    breakdown = [
      { label: 'Layers', value: fmtNum(f.num_hidden_layers || (fullLayers + linearLayers)) },
      { label: 'Gated MLA layers', value: fmtNum(fullLayers) },
      { label: 'KDA linear layers', value: fmtNum(linearLayers) },
      { label: 'KV LoRA rank', value: fmtNum(kvLoraRank) },
      { label: 'QK RoPE head dim', value: fmtNum(qkRopeHd) },
      { label: 'MLA KV elements', value: fmtNum(mlaElements) },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'KDA state included', value: includeLinear ? 'Yes' : 'No', tip: 'Whether KDA conv + recurrent state is included in the calculation.' },
      { label: 'KDA conv elements', value: fmtNum(kdaConvElements) },
      { label: 'KDA recurrent elements', value: fmtNum(kdaRecurrentElements) },
      { label: 'Per-token elements', value: fmtNum(fullLayers * (kvLoraRank + qkRopeHd)), tip: 'Gated MLA only: full_layers \u00d7 (kv_lora_rank + qk_rope_head_dim)' },
      { label: 'Total bytes', value: fmtNum(kvBytes) },
    ];

  // ── msa_gqa (MiniMax Sparse Attention + GQA) ──
  } else if (formula === 'msa_gqa') {
    const layers = f.num_hidden_layers;
    const kvHeads = f.num_key_value_heads;
    const hd = f.head_dim;

    // Main GQA KV: ALL layers store full KV (MSA sparsity is compute-only, not storage)
    const elements = 2 * layers * kvHeads * hd * tokens;
    kvBytes = elements * precB;
    perTokenBytes = 2 * layers * kvHeads * hd * precB;

    // Sparse attention index branch: only sparse layers store K_idx
    const idxHd = f.sparse_index_dim;
    const idxHeads = f.sparse_num_index_heads;
    const sparseFreq = f.sparse_attention_freq;
    const sparseLayers = sparseFreq ? sparseFreq.filter(function(v) { return v === 1; }).length : 0;
    const fullAttnLayers = layers - sparseLayers;
    const idxElements = sparseLayers * idxHeads * idxHd * tokens;
    idxBytes = idxElements * idxB;
    perTokenBytes += sparseLayers * idxHeads * idxHd * idxB;

    // MTP draft layers
    let draftLayers = 0;
    if (includeDraft && f.mtp_transformer_layers) {
      draftLayers = f.mtp_transformer_layers;
      const draftElements = 2 * draftLayers * kvHeads * hd * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += 2 * draftLayers * kvHeads * hd * precB;
    }

    formulaTitle = model.label + ' MSA sparse attention + GQA';

    // Per-layer bytes for pattern visualization
    const kPerLayer = kvHeads * hd * tokens * precB;
    const vPerLayer = kvHeads * hd * tokens * precB;
    const idxPerLayer = idxHeads * idxHd * tokens * idxB;

    var msaKvBar = [{ type: 'full', bytes: kvBytes }];
    var msaIdxBar = [{ type: 'indexer', bytes: idxBytes }];
    var msaTotalBar = [{ type: 'full', bytes: kvBytes }, { type: 'indexer', bytes: idxBytes }];
    formulas = [
      { name: 'KV', tip: 'Main GQA KV cache for all layers. MSA stores full uncompressed K/V; sparsity is compute-only.', expr: '2 \u00d7 L \u00d7 h_kv \u00d7 d_h \u00d7 T \u00d7 p', values: { L: layers, h_kv: kvHeads, d_h: hd, T: tokens, p: precB }, resultValue: kvBytes, bar: msaKvBar, ibarVal: fmtBytes(kvBytes) },
      { name: 'Idx', tip: 'Sparse index branch K_idx cache for sparse layers only. Used for block-level TopK selection.', expr: 'L_sp \u00d7 h_idx \u00d7 d_idx \u00d7 T \u00d7 p_idx', values: { L_sp: sparseLayers, h_idx: idxHeads, d_idx: idxHd, T: tokens, p_idx: idxB }, resultValue: idxBytes, bar: msaIdxBar, ibarVal: fmtBytes(idxBytes) },
      { name: 'Total', tip: 'Combined cache payload for all concurrent sequences.', expr: 'B \u00d7 (KV + Idx)', values: { KV: kvBytes, Idx: idxBytes }, resultValue: kvBytes + idxBytes, bar: msaTotalBar, ibarVal: fmtBytes(seqs * (kvBytes + idxBytes)) }
    ];

    patterns = [];
    if (fullAttnLayers > 0) {
      patterns.push({
        segs: [{ type: 'full', ratio: 0.5 }, { type: 'full-alt', ratio: 0.5 }],
        count: fullAttnLayers,
        label: 'full attn',
        bytes: kPerLayer + vPerLayer
      });
    }
    if (sparseLayers > 0) {
      var sparseTotal = kPerLayer + vPerLayer + idxPerLayer;
      patterns.push({
        segs: [
          { type: 'full', ratio: kPerLayer / sparseTotal },
          { type: 'full-alt', ratio: vPerLayer / sparseTotal },
          { type: 'indexer', ratio: idxPerLayer / sparseTotal }
        ],
        count: sparseLayers,
        label: 'sparse attn',
        bytes: sparseTotal
      });
    }
    legendTypes = ['full', 'indexer'];

    breakdown = [
      { label: 'Layers', value: fmtNum(layers) },
      { label: 'Full attention layers', value: fmtNum(fullAttnLayers) },
      { label: 'Sparse attention layers', value: fmtNum(sparseLayers) },
      { label: 'KV heads', value: fmtNum(kvHeads) },
      { label: 'Head dim', value: fmtNum(hd) },
      { label: 'KV elements', value: fmtNum(elements) },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'Index heads', value: fmtNum(idxHeads) },
      { label: 'Index head dim', value: fmtNum(idxHd) },
      { label: 'Index elements', value: fmtNum(idxElements) },
      { label: 'Index precision bytes', value: idxB.toString() },
    ];
    if (draftLayers > 0) {
      breakdown.push({ label: 'Draft layers included', value: fmtNum(draftLayers), tip: 'Extra MTP/draft layers after the main transformer layers.' });
    }
    breakdown.push({ label: 'Total bytes', value: fmtNum(kvBytes + idxBytes) });
  }

  return {
    kvBytes: kvBytes,
    idxBytes: idxBytes,
    idxLayers: idxLayers,
    mainKvBytes: mainKvBytes === null ? (draftKvBytes > 0 ? kvBytes - draftKvBytes : kvBytes) : mainKvBytes,
    draftKvBytes: draftKvBytes,
    kvLayerBytes: kvLayerBytes,
    idxLayerBytes: idxLayerBytes,
    linearLayerBytes: linearLayerBytes,
    globalCacheBytes: globalCacheBytes,
    perTokenBytes: perTokenBytes,
    breakdown: breakdown,
    formulas: formulas,
    formulaTitle: formulaTitle,
    patterns: patterns,
    legendTypes: legendTypes
  };
}

/**
 * Calculate total KV-cache size in GB (decimal) for one sequence.
 * @param {Object} model   - Model entry from MODEL_DATA
 * @param {number} tokens  - Sequence length
 * @param {number} precB   - Bytes per element for KV precision
 * @param {number} idxB    - Bytes per element for indexer precision
 * @param {Object} options - { includeDraft: bool, includeLinear: bool }
 * @returns {number} Total GB = (kvBytes + idxBytes) / 1e9
 */
function calcKvCacheGB(model, tokens, precB, idxB, options) {
  const r = calcKvCache(model, tokens, precB, idxB, options);
  return (r.kvBytes + r.idxBytes) / 1e9;
}

/**
 * Calculate KV-cache size across a range of token counts (for charting).
 *
 * @param {Object}   model       - Model entry from MODEL_DATA
 * @param {number[]} tokenPoints - Array of token counts to evaluate
 * @param {number}   precB       - Bytes per element for KV precision
 * @param {number}   idxB        - Bytes per element for indexer precision
 * @param {Object}   options     - { includeDraft: bool, includeLinear: bool }
 * @returns {Array<{tokens: number, gb: number}>}
 */
function calcKvCacheSeries(model, tokenPoints, precB, idxB, options) {
  return tokenPoints.map(function (t) {
    return { tokens: t, gb: calcKvCacheGB(model, t, precB, idxB, options) };
  });
}
