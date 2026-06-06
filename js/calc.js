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
  k_c: 'conv_kernel_dim'
};

/**
 * Format bytes using binary (IEC) prefixes (KiB, MiB, GiB).
 * @param {number} bytes
 * @returns {string}
 */
function fmtBytes(bytes) {
  if (bytes < 1024) return bytes.toFixed(0) + ' B';
  if (bytes < 1024**2) return (bytes / 1024).toFixed(5) + ' KiB';
  if (bytes < 1024**3) return (bytes / 1024**2).toFixed(5) + ' MiB';
  return (bytes / 1024**3).toFixed(5) + ' GiB';
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
 * @returns {{ kvBytes: number, idxBytes: number, perTokenBytes: number,
 *             breakdown: Array<{label:string,value:string,tip?:string}>,
 *             formulas: Array<{name:string,tip:string,expr:string}>,
 *             formulaTitle: string }}
 */
function calcKvCache(model, tokens, precB, idxB, options) {
  const f = model.fields;
  const formula = model.formula;
  const includeDraft  = !!(options && options.includeDraft);
  const includeLinear = !!(options && options.includeLinear);

  let kvBytes = 0, idxBytes = 0, perTokenBytes = 0;
  let breakdown = [];
  let formulas = [];
  let formulaTitle = '';

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
      { name: 'KV', tip: 'Main ' + model.label + ' KV cache before applying precision.', expr: '2 \u00d7 L \u00d7 h_kv \u00d7 d_h \u00d7 T \u00d7 p', values: { L: layers, h_kv: kvHeads, d_h: hd, T: tokens, p: precB }, resultValue: kvBytes },
      { name: 'Total', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'B \u00d7 KV', values: { KV: kvBytes }, resultValue: kvBytes }
    ];

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
    formulas = [
      { name: 'KV', tip: 'Compressed KV cache using MLA projection. Each layer stores (kv_lora_rank + qk_rope_head_dim) elements per token.', expr: 'L \u00d7 (d_c + d_r) \u00d7 T \u00d7 p', values: { L: layers, d_c: kvLoraRank, d_r: qkRopeHd, T: tokens, p: precB }, resultValue: kvBytes },
      { name: 'Total', tip: 'Combined cache payload for all concurrent sequences.', expr: 'B \u00d7 KV', values: { KV: kvBytes }, resultValue: kvBytes }
    ];

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
    const idxElements = layers * idxHd * tokens;
    idxBytes = idxElements * idxB;
    perTokenBytes += layers * idxHd * idxB;

    let draftLayers = 0;
    if (includeDraft && f.num_nextn_predict_layers) {
      draftLayers = f.num_nextn_predict_layers;
      const draftElements = draftLayers * (kvLoraRank + qkRopeHd) * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += draftLayers * (kvLoraRank + qkRopeHd) * precB;
    }

    formulaTitle = model.label + ' DSA + MLA attention';
    formulas = [
      { name: 'KV', tip: 'Compressed KV cache using MLA projection. Each layer stores (kv_lora_rank + qk_rope_head_dim) elements per token.', expr: 'L \u00d7 (d_c + d_r) \u00d7 T \u00d7 p', values: { L: layers, d_c: kvLoraRank, d_r: qkRopeHd, T: tokens, p: precB }, resultValue: kvBytes },
      { name: 'Idx', tip: 'Indexer cache for DSA sparse attention lookup.', expr: 'L \u00d7 d_idx \u00d7 T \u00d7 p_idx', values: { L: layers, d_idx: idxHd, T: tokens, p_idx: idxB }, resultValue: idxBytes },
      { name: 'Total', tip: 'Combined cache payload for all concurrent sequences.', expr: 'B \u00d7 (KV + Idx)', values: { KV: kvBytes, Idx: idxBytes }, resultValue: kvBytes + idxBytes }
    ];

    breakdown = [
      { label: 'Layers', value: fmtNum(layers) },
      { label: 'KV LoRA rank', value: fmtNum(kvLoraRank) },
      { label: 'QK RoPE head dim', value: fmtNum(qkRopeHd) },
      { label: 'KV elements', value: fmtNum(elements) },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'Indexer head dim', value: fmtNum(idxHd) },
      { label: 'Indexer elements', value: fmtNum(idxElements) },
      { label: 'Indexer precision bytes', value: idxB.toString() },
    ];
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
    formulas = [
      { name: 'KV_sw', tip: 'ALL layers contribute to the sliding window KV, including ratio=0 layers.', expr: 'L \u00d7 W \u00d7 d_h \u00d7 p', values: { L: totalLayers, W: sw, d_h: hd, p: precB }, resultValue: slidingElements * precB },
      { name: 'KV_cmp', tip: 'Compressed KV cache from layers whose compress_ratio is greater than zero; each layer keeps floor(tokens / compress_ratio) compressed slots.', expr: '\u03a3(r\u003e0) \u230aT/r\u230b \u00d7 d_h \u00d7 p', values: { T: tokens, d_h: hd, p: precB }, resultValue: compressedElements * precB },
      { name: 'KV', tip: 'Main ' + model.label + ' KV cache before adding the separate indexer cache.', expr: 'KV_sw + KV_cmp', values: { KV_sw: kvBytes, KV_cmp: compressedElements * precB }, resultValue: kvBytes },
      { name: 'Idx', tip: 'Ratio=4 layers keep an extra compressed indexer cache that can use a separate precision.', expr: 'L_4 \u00d7 \u230aT/4\u230b \u00d7 d_idx \u00d7 p_idx', values: { L_4: ratio4Layers, T: tokens, d_idx: idxHd, p_idx: idxB }, resultValue: idxBytes },
      { name: 'Total', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'B \u00d7 (KV + Idx)', values: { KV: totalKvBytes, Idx: idxBytes }, resultValue: totalKvBytes + idxBytes }
    ];

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
    formulas = [
      { name: 'KV_f', tip: 'Full attention layers store KV for the entire context length.', expr: 'L_f \u00d7 h_f \u00d7 (d_f + d_vf) \u00d7 T \u00d7 p', values: { L_f: fullLayers, h_f: globalKvHeads, d_f: globalHd, d_vf: fullVHd, T: tokens, p: precB }, resultValue: fullBytes },
      { name: 'KV_s', tip: 'Sliding window attention layers only store KV for the local window.', expr: 'L_s \u00d7 h_s \u00d7 (d_s + d_vs) \u00d7 min(T, W) \u00d7 p', values: { L_s: slidingLayers, h_s: slidingKvHeads, d_s: slidingHd, d_vs: slidingVHd, T: tokens, W: sw, p: precB }, resultValue: slidingElements * precB },
      { name: 'KV', tip: 'Combined KV cache for both attention types.', expr: 'KV_f + KV_s', values: { KV_f: fullBytes, KV_s: slidingElements * precB }, resultValue: kvBytes },
      { name: 'Total', tip: 'Combined cache payload for all concurrent sequences.', expr: 'B \u00d7 KV', values: { KV: kvBytes }, resultValue: kvBytes }
    ];

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
      { name: 'KV_f', tip: 'Full attention layers use standard GQA KV cache for the entire context.', expr: '2 \u00d7 L_f \u00d7 h_kv \u00d7 d_h \u00d7 T \u00d7 p', values: { L_f: fullLayers, h_kv: kvHeads, d_h: hd, T: tokens, p: precB }, resultValue: fullBytes },
      { name: 'S_conv', tip: 'Linear attention conv kernel state, fixed per sequence in BF16.', expr: 'B \u00d7 L_l \u00d7 k_c \u00d7 (2 \u00d7 h_kl \u00d7 d_kl + h_vl \u00d7 d_vl) \u00d7 2', values: { L_l: linearLayers, k_c: convDim, h_kl: linKvHeads, d_kl: linKeyHd, h_vl: linValHeads, d_vl: linValHd }, resultValue: linConvBytes },
      { name: 'S_rec', tip: 'Linear attention recurrent state, fixed per sequence in FP32.', expr: 'B \u00d7 L_l \u00d7 h_vl \u00d7 d_kl \u00d7 d_vl \u00d7 4', values: { L_l: linearLayers, h_vl: linValHeads, d_kl: linKeyHd, d_vl: linValHd }, resultValue: linRecurrentBytes },
      { name: 'Total', tip: 'Combined full + linear KV cache.', expr: 'KV_f + S_conv + S_rec', values: { KV_f: fullBytes, S_conv: linConvBytes, S_rec: linRecurrentBytes }, resultValue: kvBytes }
    ];

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
  }

  return {
    kvBytes: kvBytes,
    idxBytes: idxBytes,
    perTokenBytes: perTokenBytes,
    breakdown: breakdown,
    formulas: formulas,
    formulaTitle: formulaTitle
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
