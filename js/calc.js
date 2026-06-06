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
      { name: 'kv_bytes', tip: 'Main ' + model.label + ' KV cache before applying precision.', expr: '2 \u00d7 layers \u00d7 kv_heads \u00d7 head_dim \u00d7 tokens \u00d7 precision_bytes', values: { layers: layers, kv_heads: kvHeads, head_dim: hd, tokens: tokens, precision_bytes: precB }, resultValue: kvBytes },
      { name: 'total_bytes', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'sequences \u00d7 kv_bytes', values: { kv_bytes: kvBytes }, resultValue: kvBytes }
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
      { name: 'kv_bytes', tip: 'Compressed KV cache using MLA projection. Each layer stores (kv_lora_rank + qk_rope_head_dim) elements per token.', expr: 'layers \u00d7 (kv_lora_rank + qk_rope_head_dim) \u00d7 tokens \u00d7 precision_bytes', values: { layers: layers, kv_lora_rank: kvLoraRank, qk_rope_head_dim: qkRopeHd, tokens: tokens, precision_bytes: precB }, resultValue: kvBytes },
      { name: 'total_bytes', tip: 'Combined cache payload for all concurrent sequences.', expr: 'sequences \u00d7 kv_bytes', values: { kv_bytes: kvBytes }, resultValue: kvBytes }
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
      { name: 'kv_bytes', tip: 'Compressed KV cache using MLA projection. Each layer stores (kv_lora_rank + qk_rope_head_dim) elements per token.', expr: 'layers \u00d7 (kv_lora_rank + qk_rope_head_dim) \u00d7 tokens \u00d7 precision_bytes', values: { layers: layers, kv_lora_rank: kvLoraRank, qk_rope_head_dim: qkRopeHd, tokens: tokens, precision_bytes: precB }, resultValue: kvBytes },
      { name: 'indexer_bytes', tip: 'Indexer cache for DSA sparse attention lookup.', expr: 'layers \u00d7 index_head_dim \u00d7 tokens \u00d7 indexer_precision_bytes', values: { layers: layers, index_head_dim: idxHd, tokens: tokens, indexer_precision_bytes: idxB }, resultValue: idxBytes },
      { name: 'total_bytes', tip: 'Combined cache payload for all concurrent sequences.', expr: 'sequences \u00d7 (kv_bytes + indexer_bytes)', values: { kv_bytes: kvBytes, indexer_bytes: idxBytes }, resultValue: kvBytes + idxBytes }
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
      { name: 'sliding_kv_bytes', tip: 'ALL layers contribute to the sliding window KV, including ratio=0 layers.', expr: 'total_layers \u00d7 sliding_window \u00d7 head_dim \u00d7 precision_bytes', values: { total_layers: totalLayers, sliding_window: sw, head_dim: hd, precision_bytes: precB }, resultValue: slidingElements * precB },
      { name: 'compressed_kv_bytes', tip: 'Compressed KV cache from layers whose compress_ratio is greater than zero; each layer keeps floor(tokens / compress_ratio) compressed slots.', expr: '\u03a3 over ratio>0 layers: floor(tokens / compress_ratio) \u00d7 head_dim \u00d7 precision_bytes', values: { tokens: tokens, head_dim: hd, precision_bytes: precB }, resultValue: compressedElements * precB },
      { name: 'kv_bytes', tip: 'Main ' + model.label + ' KV cache before adding the separate indexer cache.', expr: 'sliding_kv_bytes + compressed_kv_bytes', values: { sliding_kv_bytes: kvBytes, compressed_kv_bytes: compressedElements * precB }, resultValue: kvBytes },
      { name: 'indexer_bytes', tip: 'Ratio=4 layers keep an extra compressed indexer cache that can use a separate precision.', expr: 'ratio4_layers \u00d7 floor(tokens / 4) \u00d7 index_head_dim \u00d7 indexer_precision_bytes', values: { ratio4_layers: ratio4Layers, tokens: tokens, index_head_dim: idxHd, indexer_precision_bytes: idxB }, resultValue: idxBytes },
      { name: 'total_bytes', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'sequences \u00d7 (kv_bytes + indexer_bytes)', values: { kv_bytes: totalKvBytes, indexer_bytes: idxBytes }, resultValue: totalKvBytes + idxBytes }
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
      { name: 'full_kv_bytes', tip: 'Full attention layers store KV for the entire context length.', expr: 'full_layers \u00d7 full_kv_heads \u00d7 (full_head_dim + full_v_head_dim) \u00d7 tokens \u00d7 precision_bytes', values: { full_layers: fullLayers, full_kv_heads: globalKvHeads, full_head_dim: globalHd, full_v_head_dim: fullVHd, tokens: tokens, precision_bytes: precB }, resultValue: fullBytes },
      { name: 'sliding_kv_bytes', tip: 'Sliding window attention layers only store KV for the local window.', expr: 'sliding_layers \u00d7 sliding_kv_heads \u00d7 (sliding_head_dim + sliding_v_head_dim) \u00d7 min(tokens, sliding_window) \u00d7 precision_bytes', values: { sliding_layers: slidingLayers, sliding_kv_heads: slidingKvHeads, sliding_head_dim: slidingHd, sliding_v_head_dim: slidingVHd, tokens: tokens, sliding_window: sw, precision_bytes: precB }, resultValue: slidingElements * precB },
      { name: 'kv_bytes', tip: 'Combined KV cache for both attention types.', expr: 'full_kv_bytes + sliding_kv_bytes', values: { full_kv_bytes: fullBytes, sliding_kv_bytes: slidingElements * precB }, resultValue: kvBytes },
      { name: 'total_bytes', tip: 'Combined cache payload for all concurrent sequences.', expr: 'sequences \u00d7 kv_bytes', values: { kv_bytes: kvBytes }, resultValue: kvBytes }
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
      { name: 'full_kv_bytes', tip: 'Full attention layers use standard GQA KV cache for the entire context.', expr: '2 \u00d7 full_attention_layers \u00d7 kv_heads \u00d7 head_dim \u00d7 tokens \u00d7 precision_bytes', values: { full_attention_layers: fullLayers, kv_heads: kvHeads, head_dim: hd, tokens: tokens, precision_bytes: precB }, resultValue: fullBytes },
      { name: 'linear_conv_state_bytes', tip: 'Linear attention conv kernel state, fixed per sequence in BF16.', expr: 'sequences \u00d7 linear_layers \u00d7 conv_kernel_dim \u00d7 (2 \u00d7 lin_key_heads \u00d7 lin_key_dim + lin_val_heads \u00d7 lin_val_dim) \u00d7 2', values: { linear_layers: linearLayers, conv_kernel_dim: convDim, lin_key_heads: linKvHeads, lin_key_dim: linKeyHd, lin_val_heads: linValHeads, lin_val_dim: linValHd }, resultValue: linConvBytes },
      { name: 'linear_recurrent_state_bytes', tip: 'Linear attention recurrent state, fixed per sequence in FP32.', expr: 'sequences \u00d7 linear_layers \u00d7 lin_val_heads \u00d7 lin_key_dim \u00d7 lin_val_dim \u00d7 4', values: { linear_layers: linearLayers, lin_val_heads: linValHeads, lin_key_dim: linKeyHd, lin_val_dim: linValHd }, resultValue: linRecurrentBytes },
      { name: 'total_bytes', tip: 'Combined full + linear KV cache.', expr: 'full_kv_bytes + linear_conv_state_bytes + linear_recurrent_state_bytes', values: { full_kv_bytes: fullBytes, linear_conv_state_bytes: linConvBytes, linear_recurrent_state_bytes: linRecurrentBytes }, resultValue: kvBytes }
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
