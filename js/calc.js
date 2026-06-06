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
      { name: 'kv_bytes', tip: 'Main ' + model.label + ' KV cache before applying precision.', expr: '2 \u00d7 layers \u00d7 kv_heads \u00d7 head_dim \u00d7 tokens \u00d7 precision_bytes', values: { layers: layers, kv_heads: kvHeads, head_dim: hd, tokens: tokens, precision_bytes: precB } },
      { name: 'total_bytes', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'sequences \u00d7 kv_bytes', values: { kv_bytes: kvBytes } }
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
    // MLA: KV cache stores compressed (kv_lora_rank) per head, not full heads
    // Per token per layer: 2 * kv_lora_rank (K and V compressed)
    const elements = 2 * layers * kvLoraRank * tokens;
    kvBytes = elements * precB;
    perTokenBytes = 2 * layers * kvLoraRank * precB;

    let draftLayers = 0;
    if (includeDraft && f.num_nextn_predict_layers) {
      draftLayers = f.num_nextn_predict_layers;
      const draftElements = 2 * draftLayers * kvLoraRank * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += 2 * draftLayers * kvLoraRank * precB;
    }

    formulaTitle = model.label + ' multi-head latent attention (MLA)';
    formulas = [
      { name: 'kv_bytes', tip: 'Compressed KV cache using MLA projection. Each layer stores kv_lora_rank compressed elements per token.', expr: '2 \u00d7 layers \u00d7 kv_lora_rank \u00d7 tokens \u00d7 precision_bytes', values: { layers: layers, kv_lora_rank: kvLoraRank, tokens: tokens, precision_bytes: precB } },
      { name: 'total_bytes', tip: 'Combined cache payload for all concurrent sequences.', expr: 'sequences \u00d7 kv_bytes', values: { kv_bytes: kvBytes } }
    ];

    breakdown = [
      { label: 'Layers', value: fmtNum(layers) },
      { label: 'KV LoRA rank', value: fmtNum(kvLoraRank) },
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
    const elements = 2 * layers * kvLoraRank * tokens;
    kvBytes = elements * precB;
    perTokenBytes = 2 * layers * kvLoraRank * precB;

    const idxHeads = f.index_n_heads;
    const idxHd = f.index_head_dim;
    const idxElements = 2 * layers * idxHeads * idxHd * tokens;
    idxBytes = idxElements * idxB;
    perTokenBytes += 2 * layers * idxHeads * idxHd * idxB;

    let draftLayers = 0;
    if (includeDraft && f.num_nextn_predict_layers) {
      draftLayers = f.num_nextn_predict_layers;
      const draftElements = 2 * draftLayers * kvLoraRank * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += 2 * draftLayers * kvLoraRank * precB;
    }

    formulaTitle = model.label + ' DSA + MLA attention';
    formulas = [
      { name: 'kv_bytes', tip: 'Compressed KV cache using MLA projection.', expr: '2 \u00d7 layers \u00d7 kv_lora_rank \u00d7 tokens \u00d7 precision_bytes', values: { layers: layers, kv_lora_rank: kvLoraRank, tokens: tokens, precision_bytes: precB } },
      { name: 'indexer_bytes', tip: 'Indexer cache for DSA sparse attention lookup.', expr: '2 \u00d7 layers \u00d7 index_n_heads \u00d7 index_head_dim \u00d7 tokens \u00d7 indexer_precision_bytes', values: { layers: layers, index_n_heads: idxHeads, index_head_dim: idxHd, tokens: tokens, indexer_precision_bytes: idxB } },
      { name: 'total_bytes', tip: 'Combined cache payload for all concurrent sequences.', expr: 'sequences \u00d7 (kv_bytes + indexer_bytes)', values: { kv_bytes: kvBytes, indexer_bytes: idxBytes } }
    ];

    breakdown = [
      { label: 'Layers', value: fmtNum(layers) },
      { label: 'KV LoRA rank', value: fmtNum(kvLoraRank) },
      { label: 'KV elements', value: fmtNum(elements) },
      { label: 'KV precision bytes', value: precB.toString() },
      { label: 'Indexer heads', value: fmtNum(idxHeads) },
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

    // Sliding window KV: all active layers (ratio > 0)
    const slidingElements = activeLayers * sw * hd;
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
      { name: 'sliding_kv_bytes', tip: 'Includes ratio=0 layers. Ratio=0 layers only contribute this fixed sliding-window KV and do not add compressed KV slots.', expr: 'active_layers \u00d7 sliding_window \u00d7 head_dim \u00d7 precision_bytes', values: { active_layers: activeLayers, sliding_window: sw, head_dim: hd, precision_bytes: precB } },
      { name: 'compressed_kv_bytes', tip: 'Compressed KV cache from layers whose compress_ratio is greater than zero; each layer keeps floor(tokens / compress_ratio) compressed slots.', expr: '\u03a3 over ratio>0 layers: floor(tokens / compress_ratio) \u00d7 head_dim \u00d7 precision_bytes', values: { tokens: tokens, head_dim: hd, precision_bytes: precB } },
      { name: 'kv_bytes', tip: 'Main ' + model.label + ' KV cache before adding the separate indexer cache.', expr: 'sliding_kv_bytes + compressed_kv_bytes', values: { sliding_kv_bytes: kvBytes, compressed_kv_bytes: compressedElements * precB } },
      { name: 'indexer_bytes', tip: 'Ratio=4 layers keep an extra compressed indexer cache that can use a separate precision.', expr: 'ratio4_layers \u00d7 floor(tokens / 4) \u00d7 index_head_dim \u00d7 indexer_precision_bytes', values: { ratio4_layers: ratio4Layers, tokens: tokens, index_head_dim: idxHd, indexer_precision_bytes: idxB } },
      { name: 'total_bytes', tip: 'Combined ' + model.label + ' cache payload for all concurrent sequences.', expr: 'sequences \u00d7 (kv_bytes + indexer_bytes)', values: { kv_bytes: totalKvBytes, indexer_bytes: idxBytes } }
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

    const fullElements = 2 * fullLayers * globalKvHeads * globalHd * tokens;
    const slidingElements = 2 * slidingLayers * kvHeads * hd * sw;

    const kvElements = fullElements + slidingElements;
    kvBytes = kvElements * precB;
    perTokenBytes = (2 * fullLayers * globalKvHeads * globalHd + 2 * slidingLayers * kvHeads * hd * (sw / tokens)) * precB;

    formulaTitle = model.label + ' mixed full + sliding window attention';
    formulas = [
      { name: 'full_kv_bytes', tip: 'Full attention layers store KV for the entire context length.', expr: '2 \u00d7 full_attention_layers \u00d7 global_kv_heads \u00d7 global_head_dim \u00d7 tokens \u00d7 precision_bytes', values: { full_attention_layers: fullLayers, global_kv_heads: globalKvHeads, global_head_dim: globalHd, tokens: tokens, precision_bytes: precB } },
      { name: 'sliding_kv_bytes', tip: 'Sliding window attention layers only store KV for the local window.', expr: '2 \u00d7 sliding_attention_layers \u00d7 kv_heads \u00d7 head_dim \u00d7 sliding_window \u00d7 precision_bytes', values: { sliding_attention_layers: slidingLayers, kv_heads: kvHeads, head_dim: hd, sliding_window: sw, precision_bytes: precB } },
      { name: 'kv_bytes', tip: 'Combined KV cache for both attention types.', expr: 'full_kv_bytes + sliding_kv_bytes', values: { full_kv_bytes: fullBytes, sliding_kv_bytes: slidingElements * precB } },
      { name: 'total_bytes', tip: 'Combined cache payload for all concurrent sequences.', expr: 'sequences \u00d7 kv_bytes', values: { kv_bytes: kvBytes } }
    ];

    breakdown = [
      { label: 'Full attention layers', value: fmtNum(fullLayers) },
      { label: 'Sliding attention layers', value: fmtNum(slidingLayers) },
      { label: 'Global KV heads', value: fmtNum(globalKvHeads) },
      { label: 'Global head dim', value: fmtNum(globalHd) },
      { label: 'Sliding KV heads', value: fmtNum(kvHeads) },
      { label: 'Sliding head dim', value: fmtNum(hd) },
      { label: 'Sliding window', value: fmtNum(sw) },
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

    const fullElements = 2 * fullLayers * kvHeads * hd * tokens;
    const fullBytes = fullElements * precB;

    const linKeyElements = linKvHeads * linKeyHd * (convDim + tokens);
    const linValElements = linValHeads * linValHd * (convDim + tokens);
    const linElements = linearLayers * (linKeyElements + linValElements);
    const linBytes = includeLinear ? linElements * precB : 0;

    kvBytes = fullBytes + linBytes;
    perTokenBytes = (2 * fullLayers * kvHeads * hd + (includeLinear ? linearLayers * (linKvHeads * linKeyHd + linValHeads * linValHd) : 0)) * precB;

    var draftLayers = 0;
    if (includeDraft && f.mtp_num_hidden_layers) {
      draftLayers = f.mtp_num_hidden_layers;
      const draftElements = 2 * draftLayers * kvHeads * hd * tokens;
      kvBytes += draftElements * precB;
      perTokenBytes += 2 * draftLayers * kvHeads * hd * precB;
    }

    formulaTitle = model.label + ' linear + full attention hybrid';
    formulas = [
      { name: 'full_kv_bytes', tip: 'Full attention layers use standard GQA KV cache for the entire context.', expr: '2 \u00d7 full_attention_layers \u00d7 kv_heads \u00d7 head_dim \u00d7 tokens \u00d7 precision_bytes', values: { full_attention_layers: fullLayers, kv_heads: kvHeads, head_dim: hd, tokens: tokens, precision_bytes: precB } },
      { name: 'linear_kv_bytes', tip: 'Linear attention layers store a fixed-size conv kernel plus per-token KV.', expr: 'linear_layers \u00d7 (lin_key_heads \u00d7 lin_key_hd + lin_val_heads \u00d7 lin_val_hd) \u00d7 (conv_kernel_dim + tokens) \u00d7 precision_bytes', values: { linear_layers: linearLayers, lin_key_heads: linKvHeads, lin_key_hd: linKeyHd, lin_val_heads: linValHeads, lin_val_hd: linValHd, conv_kernel_dim: convDim, tokens: tokens, precision_bytes: precB } },
      { name: 'kv_bytes', tip: 'Combined full + linear KV cache.', expr: 'full_kv_bytes + linear_kv_bytes', values: { full_kv_bytes: fullBytes, linear_kv_bytes: linBytes } },
      { name: 'total_bytes', tip: 'Combined cache payload for all concurrent sequences.', expr: 'sequences \u00d7 kv_bytes', values: { kv_bytes: kvBytes } }
    ];

    breakdown = [
      { label: 'Full attention layers', value: fmtNum(fullLayers) },
      { label: 'Linear attention layers', value: fmtNum(linearLayers) },
      { label: 'Full KV heads', value: fmtNum(kvHeads) },
      { label: 'Full head dim', value: fmtNum(hd) },
      { label: 'Full KV elements', value: fmtNum(fullElements) },
      { label: 'Linear key heads', value: fmtNum(linKvHeads) },
      { label: 'Linear value heads', value: fmtNum(linValHeads) },
      { label: 'Linear key head dim', value: fmtNum(linKeyHd) },
      { label: 'Linear value head dim', value: fmtNum(linValHd) },
      { label: 'Conv kernel dim', value: fmtNum(convDim) },
      { label: 'Linear KV elements', value: fmtNum(linElements) },
      { label: 'KV precision bytes', value: precB.toString() },
    ];
    if (draftLayers > 0) {
      breakdown.push({ label: 'Draft layers included', value: fmtNum(draftLayers), tip: 'Extra MTP/draft layers after the main transformer layers.' });
    }
    breakdown.push({ label: 'Total bytes', value: fmtNum(kvBytes) });
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
