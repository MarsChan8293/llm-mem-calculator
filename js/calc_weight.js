var WEIGHT_SYMBOL_NAMES = {
  h: 'hidden_size', V: 'vocab_size',
  n_q: 'num_attention_heads',
  d_v: 'v_head_dim',
  q_r: 'q_lora_rank', o_r: 'o_lora_rank',
  qk: 'qk_head_dim', qk_nope: 'qk_nope_head_dim',
  I: 'intermediate_size', I_m: 'moe_intermediate_size',
  I_s: 'shared_expert_intermediate_size',
  N_e: 'n_routed_experts', N_s: 'n_shared_experts',
  L_d: 'dense_ffn_layers', L_m: 'moe_ffn_layers',
  L_idx: 'indexer_source_layers', h_idx: 'indexer_heads',
  d_idx: 'indexer_head_dim', k_pool: 'indexer_pool_size',
  h_idx_kv: 'indexer_kv_heads', qkv: 'linear_qkv_width',
  h_l: 'linear_num_heads', d_l: 'linear_head_dim', g: 'o_groups',
};

var WEIGHT_BAR_COLOR_MAP = {
  'attn': 'seg-full', 'ffn-dense': 'seg-full-alt',
  'ffn-shared': 'seg-compressed', 'ffn-expert': 'seg-indexer',
  'embed': 'seg-rope', 'vision': 'seg-fixed-alt',
};
var WEIGHT_BAR_HEX_MAP = {
  'attn': '#4263eb', 'ffn-dense': '#f59e0b',
  'ffn-shared': '#e67700', 'ffn-expert': '#e03131',
  'embed': '#9c36b5', 'vision': '#495057',
};
var WEIGHT_LEGEND_LABEL_MAP = {
  'attn': 'Attention', 'ffn-dense': 'Dense FFN',
  'ffn-shared': 'Shared Expert', 'ffn-expert': 'Routed Experts',
  'embed': 'Embedding', 'vision': 'Vision Tower',
};

function fmtWBytes(bytes) {
  return (bytes / 1e9).toFixed(5) + ' GB';
}

function fmtWNum(n) { return n.toLocaleString('en-US'); }

function isMoeLayer(wf, layerIndex) {
  if (!wf) return false;
  if (wf.moe_layer_freq && Array.isArray(wf.moe_layer_freq)) {
    return wf.moe_layer_freq[layerIndex] === 1;
  }
  if (wf.first_k_dense_replace != null) {
    return layerIndex >= wf.first_k_dense_replace;
  }
  if (!wf.n_routed_experts) return false;
  return true;
}

function calcWeight(model, wtPrecB) {
  var f = model.fields;
  var wf = model.weight_fields || {};
  var formula = model.formula;
  var L = f.num_hidden_layers;
  var h = wf.hidden_size || 0;
  var V = wf.vocab_size || 0;

  var attnParams = 0;
  var ffnDenseParams = 0;
  var ffnSharedParams = 0;
  var ffnExpertParams = 0;
  var embedParams = 0;
  var visionParams = (model.vision_fields && model.vision_fields.params) || 0;
  // FP8 resident Engram hash tables, plus one scale byte per 32 values.
  var engramParams = Array.isArray(f.engram_num_embeddings) ? f.engram_num_embeddings.reduce(function(n, rows) { return n + rows * (f.engram_head_dim || 0); }, 0) : 0;
  var engramBytes = engramParams * (1 + 1 / 32);

  var breakdown = [];
  var formulas = [];
  var formulaTitle = '';
  var patterns = [];
  var legendTypes = [];

  var ffnMats = (wf.ffn_type === 'swiglu') ? 3 : 2;

  var n_q = wf.num_attention_heads || f.num_attention_heads || 0;
  var h_kv = f.num_key_value_heads || 0;
  var d_h = f.head_dim || 0;
  var d_v = wf.v_head_dim || f.v_head_dim || d_h;

  if (formula === 'standard_gqa') {
    var Wq = h * (n_q * d_h);
    var Wk = h * (h_kv * d_h);
    var Wv = h * (h_kv * d_v);
    var Wo = (n_q * d_v) * h;
    var attnPerLayer = Wq + Wk + Wv + Wo;

    var I = wf.intermediate_size || 0;
    var denseFfnPerLayer = ffnMats * h * I;

    var nRouted = wf.n_routed_experts || 0;
    var nShared = wf.n_shared_experts || 0;
    var Im = wf.moe_intermediate_size || I;
    var Is = wf.shared_expert_intermediate_size || Im;

    var sharedPerLayer = nShared * ffnMats * h * Is;
    var expertPerLayer = nRouted * ffnMats * h * Im;

    var denseLayerCount = 0;
    var moeLayerCount = 0;
    for (var i = 0; i < L; i++) {
      if (isMoeLayer(wf, i)) { moeLayerCount++; } else { denseLayerCount++; }
    }

    attnParams = L * attnPerLayer;
    ffnDenseParams = denseLayerCount * denseFfnPerLayer;
    ffnSharedParams = moeLayerCount * sharedPerLayer;
    ffnExpertParams = moeLayerCount * expertPerLayer;

    var tieEmbed = wf.tie_word_embeddings;
    embedParams = tieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + ' standard GQA';
    formulas = [
      { name: 'Attn', tip: 'Attention weights per layer: Q + K + V + O projections.', expr: 'h\u00d7n_q\u00d7d_h + 2\u00d7h\u00d7h_kv\u00d7d_h + n_q\u00d7d_h\u00d7h', values: { h: h, n_q: n_q, h_kv: h_kv, d_h: d_h, d_v: d_v }, resultValue: attnPerLayer, bar: [{ type: 'attn', bytes: attnPerLayer * wtPrecB }], ibarVal: fmtWNum(attnPerLayer) },
      { name: 'FFN_d', tip: 'Dense FFN per layer (' + ffnMats + ' matrices).', expr: ffnMats + '\u00d7h\u00d7I', values: { h: h, I: I }, resultValue: denseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(denseFfnPerLayer) }
    ];
    if (nRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s\u00d7' + ffnMats + '\u00d7h\u00d7I_s', values: { N_s: nShared, h: h, I_s: Is }, resultValue: sharedPerLayer, bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * wtPrecB }], ibarVal: fmtWNum(sharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed experts FFN per MoE layer.', expr: 'N_e\u00d7' + ffnMats + '\u00d7h\u00d7I_m', values: { N_e: nRouted, h: h, I_m: Im }, resultValue: expertPerLayer, bar: [{ type: 'ffn-expert', bytes: expertPerLayer * wtPrecB }], ibarVal: fmtWNum(expertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: tieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: tieEmbed ? 'V\u00d7h' : '2\u00d7V\u00d7h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    patterns = [];
    if (denseLayerCount > 0) {
      var denseTotal = attnPerLayer + denseFfnPerLayer;
      patterns.push({
        segs: [{ type: 'attn', ratio: attnPerLayer / denseTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / denseTotal }],
        count: denseLayerCount,
        label: 'dense FFN',
        bytes: denseTotal * wtPrecB
      });
    }
    if (moeLayerCount > 0) {
      var moeTotal = attnPerLayer + sharedPerLayer + expertPerLayer;
      var moeSegs = [{ type: 'attn', ratio: attnPerLayer / moeTotal }];
      if (nShared > 0) moeSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / moeTotal });
      moeSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / moeTotal });
      patterns.push({
        segs: moeSegs,
        count: moeLayerCount,
        label: 'MoE FFN',
        bytes: moeTotal * wtPrecB
      });
    }
    legendTypes = nRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads (Q)', value: fmtWNum(n_q) },
      { label: 'KV heads', value: fmtWNum(h_kv) },
      { label: 'Head dim', value: fmtWNum(d_h) },
      { label: 'V head dim', value: fmtWNum(d_v) },
      { label: 'Attention per layer', value: fmtWNum(attnPerLayer) },
    ];
    if (denseLayerCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(denseLayerCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(denseFfnPerLayer) });
    }
    if (moeLayerCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(moeLayerCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(nRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(nShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(Im) });
      if (nShared > 0) breakdown.push({ label: 'Shared expert intermediate size', value: fmtWNum(Is) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(sharedPerLayer) });
      breakdown.push({ label: 'Routed experts per layer', value: fmtWNum(expertPerLayer) });
    }
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: tieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'mla' || formula === 'dsa_mla') {
    var qLoraRank = wf.q_lora_rank || 0;
    var qkRopeHd = f.qk_rope_head_dim || 0;
    var qkNopeHd = f.qk_nope_head_dim || 0;
    var kvLoraRank = f.kv_lora_rank || 0;
    var qkHd = f.qk_head_dim || (qkNopeHd + qkRopeHd);

    var Wqa = h * qLoraRank;
    var Wqb = qLoraRank * (n_q * qkHd);
    var Wkva = h * (kvLoraRank + qkRopeHd);
    var Wkvb = kvLoraRank * n_q * (qkNopeHd + d_v);
    var Wo_mla = (n_q * d_v) * h;
    var attnPerLayer = Wqa + Wqb + Wkva + Wkvb + Wo_mla;

    var idxHd = f.index_head_dim || 0;
    var idxHeads = f.index_n_heads || 0;
    var idxPerLayer = (formula === 'dsa_mla' && idxHeads > 0) ? h * (idxHeads * idxHd) : 0;
    var idxParams = idxPerLayer > 0 ? L * idxPerLayer : 0;

    var I = wf.intermediate_size || 0;
    var denseFfnPerLayer = ffnMats * h * I;

    var nRouted = wf.n_routed_experts || 0;
    var nShared = wf.n_shared_experts || 0;
    var Im = wf.moe_intermediate_size || I;
    var Is = wf.shared_expert_intermediate_size || Im;

    var sharedPerLayer = nShared * ffnMats * h * Is;
    var expertPerLayer = nRouted * ffnMats * h * Im;

    var denseLayerCount = 0;
    var moeLayerCount = 0;
    for (var i = 0; i < L; i++) {
      if (isMoeLayer(wf, i)) { moeLayerCount++; } else { denseLayerCount++; }
    }

    attnParams = L * attnPerLayer + idxParams;
    ffnDenseParams = denseLayerCount * denseFfnPerLayer;
    ffnSharedParams = moeLayerCount * sharedPerLayer;
    ffnExpertParams = moeLayerCount * expertPerLayer;

    var tieEmbed = wf.tie_word_embeddings;
    embedParams = tieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + (formula === 'dsa_mla' ? ' DSA+MLA' : ' MLA');
    formulas = [
      { name: 'Attn', tip: 'MLA attention: Q LoRA down/up + KV LoRA down/up + O projection.', expr: 'h\u00d7q_r + q_r\u00d7n_q\u00d7qk + h\u00d7(d_c+d_r) + d_c\u00d7n_q\u00d7(qk_nope+d_v) + n_q\u00d7d_v\u00d7h', values: { h: h, q_r: qLoraRank, n_q: n_q, qk: qkHd, d_c: kvLoraRank, d_r: qkRopeHd, qk_nope: qkNopeHd, d_v: d_v }, resultValue: attnPerLayer, bar: [{ type: 'attn', bytes: attnPerLayer * wtPrecB }], ibarVal: fmtWNum(attnPerLayer) },
    ];
    if (idxPerLayer > 0) {
      formulas.push({ name: 'Idx', tip: 'Sparse index key projection per layer.', expr: 'h\u00d7h_idx\u00d7d_idx', values: { h: h, h_idx: idxHeads, d_idx: idxHd }, resultValue: idxPerLayer, bar: [{ type: 'attn', bytes: idxPerLayer * wtPrecB }], ibarVal: fmtWNum(idxPerLayer) });
    }
    if (denseLayerCount > 0) {
      formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer (' + ffnMats + ' matrices).', expr: ffnMats + '\u00d7h\u00d7I', values: { h: h, I: I }, resultValue: denseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(denseFfnPerLayer) });
    }
    if (nRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s\u00d7' + ffnMats + '\u00d7h\u00d7I_s', values: { N_s: nShared, h: h, I_s: Is }, resultValue: sharedPerLayer, bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * wtPrecB }], ibarVal: fmtWNum(sharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed experts FFN per MoE layer.', expr: 'N_e\u00d7' + ffnMats + '\u00d7h\u00d7I_m', values: { N_e: nRouted, h: h, I_m: Im }, resultValue: expertPerLayer, bar: [{ type: 'ffn-expert', bytes: expertPerLayer * wtPrecB }], ibarVal: fmtWNum(expertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: tieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: tieEmbed ? 'V\u00d7h' : '2\u00d7V\u00d7h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    patterns = [];
    if (denseLayerCount > 0) {
      var denseTotal = attnPerLayer + denseFfnPerLayer;
      patterns.push({
        segs: [{ type: 'attn', ratio: attnPerLayer / denseTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / denseTotal }],
        count: denseLayerCount,
        label: 'dense FFN',
        bytes: denseTotal * wtPrecB
      });
    }
    if (moeLayerCount > 0) {
      var moeTotal = attnPerLayer + sharedPerLayer + expertPerLayer;
      var moeSegs = [{ type: 'attn', ratio: attnPerLayer / moeTotal }];
      if (nShared > 0) moeSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / moeTotal });
      moeSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / moeTotal });
      patterns.push({
        segs: moeSegs,
        count: moeLayerCount,
        label: 'MoE FFN',
        bytes: moeTotal * wtPrecB
      });
    }
    legendTypes = nRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads', value: fmtWNum(n_q) },
      { label: 'KV LoRA rank', value: fmtWNum(kvLoraRank) },
      { label: 'Q LoRA rank', value: fmtWNum(qLoraRank) },
      { label: 'QK head dim', value: fmtWNum(qkHd) },
      { label: 'QK nope head dim', value: fmtWNum(qkNopeHd) },
      { label: 'QK RoPE head dim', value: fmtWNum(qkRopeHd) },
      { label: 'V head dim', value: fmtWNum(d_v) },
      { label: 'Attention per layer', value: fmtWNum(attnPerLayer) },
    ];
    if (idxPerLayer > 0) {
      breakdown.push({ label: 'Index heads', value: fmtWNum(idxHeads) });
      breakdown.push({ label: 'Index head dim', value: fmtWNum(idxHd) });
      breakdown.push({ label: 'Index projection per layer', value: fmtWNum(idxPerLayer) });
    }
    if (denseLayerCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(denseLayerCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(denseFfnPerLayer) });
    }
    if (moeLayerCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(moeLayerCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(nRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(nShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(Im) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(sharedPerLayer) });
      breakdown.push({ label: 'Routed experts per layer', value: fmtWNum(expertPerLayer) });
    }
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: tieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'deepseek_v4_hybrid') {
    var qLoraRank = wf.q_lora_rank || 0;
    var oLoraRank = wf.o_lora_rank || 0;
    var oGroups = wf.o_groups || 1;
    var hd = d_h;

    var Wqa = h * qLoraRank;
    var Wqb = qLoraRank * (n_q * hd);
    var Wk = h * hd;
    var Wv = h * hd;
    var WoDown = (n_q * hd) * (oLoraRank * oGroups);
    var WoUp = (oLoraRank * oGroups) * h;
    var attnPerLayer = Wqa + Wqb + Wk + Wv + WoDown + WoUp;

    var idxHd = f.index_head_dim || 0;
    var idxHeads = f.index_n_heads || 0;
    var compressRatios = f.compress_ratios;
    var idxLayerCount = 0;
    if (compressRatios) {
      for (var ci = 0; ci < compressRatios.length; ci++) {
        if (compressRatios[ci] > 0) idxLayerCount++;
      }
    } else if (idxHeads > 0) {
      idxLayerCount = L;
    }
    var idxPerLayer = idxHeads > 0 ? h * (idxHeads * idxHd) : 0;
    var idxParams = idxLayerCount * idxPerLayer;

    var nRouted = wf.n_routed_experts || 0;
    var nShared = wf.n_shared_experts || 0;
    var Im = wf.moe_intermediate_size || 0;
    var Is = wf.shared_expert_intermediate_size || Im;

    var sharedPerLayer = nShared * ffnMats * h * Is;
    var expertPerLayer = nRouted * ffnMats * h * Im;

    var denseLayerCount = 0;
    var moeLayerCount = 0;
    for (var i = 0; i < L; i++) {
      if (isMoeLayer(wf, i)) { moeLayerCount++; } else { denseLayerCount++; }
    }

    var I = wf.intermediate_size || 0;
    var denseFfnPerLayer = ffnMats * h * I;

    attnParams = L * attnPerLayer + idxParams;
    ffnDenseParams = denseLayerCount * denseFfnPerLayer;
    ffnSharedParams = moeLayerCount * sharedPerLayer;
    ffnExpertParams = moeLayerCount * expertPerLayer;

    var tieEmbed = wf.tie_word_embeddings;
    embedParams = tieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + ' hybrid attention';
    formulas = [
      { name: 'Attn', tip: 'V4 attention: Q LoRA + K/V direct + O grouped LoRA.', expr: 'h\u00d7q_r + q_r\u00d7n_q\u00d7d_h + 2\u00d7h\u00d7d_h + n_q\u00d7d_h\u00d7(o_r\u00d7g) + o_r\u00d7g\u00d7h', values: { h: h, q_r: qLoraRank, n_q: n_q, d_h: hd, o_r: oLoraRank, g: oGroups }, resultValue: attnPerLayer, bar: [{ type: 'attn', bytes: attnPerLayer * wtPrecB }], ibarVal: fmtWNum(attnPerLayer) },
    ];
    if (idxLayerCount > 0 && idxPerLayer > 0) {
      formulas.push({ name: 'Idx', tip: 'Sparse index key projection per index layer.', expr: 'h\u00d7h_idx\u00d7d_idx', values: { h: h, h_idx: idxHeads, d_idx: idxHd }, resultValue: idxPerLayer, bar: [{ type: 'attn', bytes: idxPerLayer * wtPrecB }], ibarVal: fmtWNum(idxPerLayer) });
    }
    if (denseLayerCount > 0) {
      formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer.', expr: ffnMats + '\u00d7h\u00d7I', values: { h: h, I: I }, resultValue: denseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(denseFfnPerLayer) });
    }
    formulas.push(
      { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s\u00d7' + ffnMats + '\u00d7h\u00d7I_s', values: { N_s: nShared, h: h, I_s: Is }, resultValue: sharedPerLayer, bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * wtPrecB }], ibarVal: fmtWNum(sharedPerLayer) },
      { name: 'FFN_e', tip: 'Routed experts FFN per MoE layer.', expr: 'N_e\u00d7' + ffnMats + '\u00d7h\u00d7I_m', values: { N_e: nRouted, h: h, I_m: Im }, resultValue: expertPerLayer, bar: [{ type: 'ffn-expert', bytes: expertPerLayer * wtPrecB }], ibarVal: fmtWNum(expertPerLayer) }
    );
    formulas.push({ name: 'Embed', tip: tieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: tieEmbed ? 'V\u00d7h' : '2\u00d7V\u00d7h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    patterns = [];
    if (denseLayerCount > 0) {
      var denseTotal = attnPerLayer + denseFfnPerLayer;
      patterns.push({
        segs: [{ type: 'attn', ratio: attnPerLayer / denseTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / denseTotal }],
        count: denseLayerCount,
        label: 'dense FFN',
        bytes: denseTotal * wtPrecB
      });
    }
    if (moeLayerCount > 0) {
      var moeTotal = attnPerLayer + sharedPerLayer + expertPerLayer;
      var moeSegs = [{ type: 'attn', ratio: attnPerLayer / moeTotal }];
      if (nShared > 0) moeSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / moeTotal });
      moeSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / moeTotal });
      patterns.push({
        segs: moeSegs,
        count: moeLayerCount,
        label: 'MoE FFN',
        bytes: moeTotal * wtPrecB
      });
    }
    legendTypes = denseLayerCount > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-shared', 'ffn-expert', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads', value: fmtWNum(n_q) },
      { label: 'Head dim', value: fmtWNum(hd) },
      { label: 'Q LoRA rank', value: fmtWNum(qLoraRank) },
      { label: 'O LoRA rank', value: fmtWNum(oLoraRank) },
      { label: 'O groups', value: fmtWNum(oGroups) },
      { label: 'Attention per layer', value: fmtWNum(attnPerLayer) },
    ];
    if (idxLayerCount > 0 && idxPerLayer > 0) {
      breakdown.push({ label: 'Index layers', value: fmtWNum(idxLayerCount) });
      breakdown.push({ label: 'Index heads', value: fmtWNum(idxHeads) });
      breakdown.push({ label: 'Index head dim', value: fmtWNum(idxHd) });
      breakdown.push({ label: 'Index projection per layer', value: fmtWNum(idxPerLayer) });
    }
    if (denseLayerCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(denseLayerCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(denseFfnPerLayer) });
    }
    breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(moeLayerCount) });
    breakdown.push({ label: 'Routed experts', value: fmtWNum(nRouted) });
    breakdown.push({ label: 'Shared experts', value: fmtWNum(nShared) });
    breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(Im) });
    if (nShared > 0) breakdown.push({ label: 'Shared expert intermediate size', value: fmtWNum(Is) });
    breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(sharedPerLayer) });
    breakdown.push({ label: 'Routed experts per layer', value: fmtWNum(expertPerLayer) });
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: tieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'deepseek_v41') {
    // DeepSeek V4.1 CED attention. The compressed KV and indexer caches are
    // source-owned at runtime, but their projection weights are counted once
    // per source layer here rather than once per consuming layer.
    var v41QLoraRank = wf.q_lora_rank || 0;
    var v41OLoraRank = wf.o_lora_rank || 0;
    var v41OGroups = wf.o_groups || 1;
    var v41Hd = d_h;
    var v41Wqa = h * v41QLoraRank;
    var v41Wqb = v41QLoraRank * (n_q * v41Hd);
    var v41Wkv = h * v41Hd;
    var v41WoDown = (n_q * v41Hd / v41OGroups) * (v41OGroups * v41OLoraRank);
    var v41WoUp = (v41OGroups * v41OLoraRank) * h;
    var v41AttnPerLayer = v41Wqa + v41Wqb + v41Wkv + v41WoDown + v41WoUp;

    var v41IdxHd = f.index_head_dim || 0;
    var v41IdxHeads = f.index_n_heads || 0;
    var v41IdxPool = f.index_kpool || 1;
    var v41IndexSources = Array.isArray(f.index_source_layer_ids) ? f.index_source_layer_ids.length : 0;
    var v41IdxPerSource = (v41QLoraRank * v41IdxHeads * v41IdxHd) + (h * v41IdxHd) + (h * v41IdxHeads) + (h * v41IdxHd) + (v41IdxHd * v41IdxPool);
    var v41IdxParams = v41IndexSources * v41IdxPerSource;

    var v41NRouted = wf.n_routed_experts || 0;
    var v41NShared = wf.n_shared_experts || 0;
    var v41Im = wf.moe_intermediate_size || 0;
    var v41Is = wf.shared_expert_intermediate_size || v41Im;
    var v41SharedPerLayer = v41NShared * ffnMats * h * v41Is;
    var v41ExpertPerLayer = v41NRouted * ffnMats * h * v41Im;
    var v41I = wf.intermediate_size || 0;
    var v41DenseFfnPerLayer = ffnMats * h * v41I;
    var v41DenseLayerCount = 0, v41MoeLayerCount = 0;
    for (var v41LayerIndex = 0; v41LayerIndex < L; v41LayerIndex++) {
      if (isMoeLayer(wf, v41LayerIndex)) v41MoeLayerCount++; else v41DenseLayerCount++;
    }

    attnParams = L * v41AttnPerLayer + v41IdxParams;
    ffnDenseParams = v41DenseLayerCount * v41DenseFfnPerLayer;
    ffnSharedParams = v41MoeLayerCount * v41SharedPerLayer;
    ffnExpertParams = v41MoeLayerCount * v41ExpertPerLayer;
    var v41TieEmbed = wf.tie_word_embeddings;
    embedParams = v41TieEmbed ? (V * h) : (2 * V * h);

    var v41EngramParams = engramParams;

    formulaTitle = model.label + ' CED attention';
    formulas = [
      { name: 'Attn', tip: 'CED attention per backbone layer: Q LoRA + one latent KV projection + grouped O LoRA.', expr: 'h×q_r + q_r×n_q×d_h + h×d_h + n_q×d_h×o_r + o_r×g×h', values: { h: h, q_r: v41QLoraRank, n_q: n_q, d_h: v41Hd, o_r: v41OLoraRank, g: v41OGroups }, resultValue: v41AttnPerLayer, bar: [{ type: 'attn', bytes: v41AttnPerLayer * wtPrecB }], ibarVal: fmtWNum(v41AttnPerLayer) },
      { name: 'Idx', tip: 'Indexer projection weights are allocated only at index_source_layer_ids.', expr: 'L_idx × (q_r×h_idx×d_idx + h×d_idx + h×h_idx + h×d_idx + d_idx×k_pool)', values: { L_idx: v41IndexSources, q_r: v41QLoraRank, h_idx: v41IdxHeads, d_idx: v41IdxHd, h: h, k_pool: v41IdxPool }, resultValue: v41IdxParams, bar: [{ type: 'attn', bytes: v41IdxParams * wtPrecB }], ibarVal: fmtWNum(v41IdxParams) },
    ];
    if (v41DenseLayerCount > 0) formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer.', expr: ffnMats + '×h×I', values: { h: h, I: v41I }, resultValue: v41DenseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: v41DenseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(v41DenseFfnPerLayer) });
    if (v41NRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s×' + ffnMats + '×h×I_s', values: { N_s: v41NShared, h: h, I_s: v41Is }, resultValue: v41SharedPerLayer, bar: [{ type: 'ffn-shared', bytes: v41SharedPerLayer * wtPrecB }], ibarVal: fmtWNum(v41SharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed expert FFN per MoE layer.', expr: 'N_e×' + ffnMats + '×h×I_m', values: { N_e: v41NRouted, h: h, I_m: v41Im }, resultValue: v41ExpertPerLayer, bar: [{ type: 'ffn-expert', bytes: v41ExpertPerLayer * wtPrecB }], ibarVal: fmtWNum(v41ExpertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: v41TieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: v41TieEmbed ? 'V×h' : '2×V×h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    var v41AttnWithSource = v41AttnPerLayer + (L > 0 ? v41IdxParams / L : 0);
    patterns = [];
    if (v41DenseLayerCount > 0) {
      var v41DenseTotal = v41AttnWithSource + v41DenseFfnPerLayer;
      patterns.push({ segs: [{ type: 'attn', ratio: v41AttnWithSource / v41DenseTotal }, { type: 'ffn-dense', ratio: v41DenseFfnPerLayer / v41DenseTotal }], count: v41DenseLayerCount, label: 'CED + dense', bytes: v41DenseTotal * wtPrecB });
    }
    if (v41MoeLayerCount > 0) {
      var v41MoeTotal = v41AttnWithSource + v41SharedPerLayer + v41ExpertPerLayer;
      var v41MoeSegs = [{ type: 'attn', ratio: v41AttnWithSource / v41MoeTotal }];
      if (v41NShared > 0) v41MoeSegs.push({ type: 'ffn-shared', ratio: v41SharedPerLayer / v41MoeTotal });
      v41MoeSegs.push({ type: 'ffn-expert', ratio: v41ExpertPerLayer / v41MoeTotal });
      patterns.push({ segs: v41MoeSegs, count: v41MoeLayerCount, label: 'CED + MoE', bytes: v41MoeTotal * wtPrecB });
    }
    legendTypes = v41NRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'CED backbone layers', value: fmtWNum(L) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads', value: fmtWNum(n_q) },
      { label: 'Head dim', value: fmtWNum(v41Hd) },
      { label: 'Q LoRA rank', value: fmtWNum(v41QLoraRank) },
      { label: 'O LoRA rank', value: fmtWNum(v41OLoraRank) },
      { label: 'O groups', value: fmtWNum(v41OGroups) },
      { label: 'Attention per layer', value: fmtWNum(v41AttnPerLayer) },
      { label: 'Indexer source layers', value: fmtWNum(v41IndexSources) },
      { label: 'Indexer params per source', value: fmtWNum(v41IdxPerSource) },
    ];
    if (v41DenseLayerCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(v41DenseLayerCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(v41DenseFfnPerLayer) });
    }
    if (v41MoeLayerCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(v41MoeLayerCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(v41NRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(v41NShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(v41Im) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(v41SharedPerLayer) });
      breakdown.push({ label: 'Routed expert per layer', value: fmtWNum(v41ExpertPerLayer) });
    }
    if (v41EngramParams > 0) breakdown.push({ label: 'Engram table params (CPU offload default)', value: fmtWNum(v41EngramParams), tip: 'FP8 hash table plus one scale byte per 32 values; counted as CPU RAM by default, not GPU VRAM.' });
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: v41TieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'mixed_full_sliding_gqa') {
    var fullLayers = f.full_attention_layers || 0;
    var slidingLayers = f.sliding_attention_layers || 0;
    var globalHd = f.global_head_dim || d_h;
    var globalKvHeads = f.num_global_key_value_heads || h_kv;
    var fullVHd = f.full_v_head_dim || f.v_head_dim || globalHd;
    var n_q_full = f.num_attention_heads || n_q;

    var slidingHd = f.sliding_head_dim || f.swa_head_dim || d_h;
    var slidingKvHeads = f.sliding_num_key_value_heads || f.swa_num_key_value_heads || h_kv;
    var slidingVHd = f.sliding_v_head_dim || f.swa_v_head_dim || slidingHd;
    var n_q_swa = f.swa_num_attention_heads || n_q;

    var fullAttnPerLayer = h * (n_q_full * globalHd) + h * (globalKvHeads * globalHd) + h * (globalKvHeads * fullVHd) + (n_q_full * fullVHd) * h;
    var slidingAttnPerLayer = h * (n_q_swa * slidingHd) + h * (slidingKvHeads * slidingHd) + h * (slidingKvHeads * slidingVHd) + (n_q_swa * slidingVHd) * h;

    var I = wf.intermediate_size || 0;
    var denseFfnPerLayer = ffnMats * h * I;

    var nRouted = wf.n_routed_experts || 0;
    var nShared = wf.n_shared_experts || 0;
    var Im = wf.moe_intermediate_size || I;
    var Is = wf.shared_expert_intermediate_size || Im;

    var sharedPerLayer = nShared * ffnMats * h * Is;
    var expertPerLayer = nRouted * ffnMats * h * Im;

    var fullDenseCount = 0, fullMoeCount = 0, slidingDenseCount = 0, slidingMoeCount = 0;
    for (var i = 0; i < L; i++) {
      var isMoe = isMoeLayer(wf, i);
      if (i < fullLayers) {
        if (isMoe) fullMoeCount++; else fullDenseCount++;
      } else {
        if (isMoe) slidingMoeCount++; else slidingDenseCount++;
      }
    }

    attnParams = fullLayers * fullAttnPerLayer + slidingLayers * slidingAttnPerLayer;
    ffnDenseParams = (fullDenseCount + slidingDenseCount) * denseFfnPerLayer;
    ffnSharedParams = (fullMoeCount + slidingMoeCount) * sharedPerLayer;
    ffnExpertParams = (fullMoeCount + slidingMoeCount) * expertPerLayer;

    var tieEmbed = wf.tie_word_embeddings;
    embedParams = tieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + ' mixed full + sliding attention';
    formulas = [
      { name: 'Attn_f', tip: 'Full attention layer: Q + K + V + O projections.', expr: 'h\u00d7(n_q_f\u00d7d_f) + h\u00d7(h_f\u00d7d_f) + h\u00d7(h_f\u00d7d_vf) + n_q_f\u00d7d_vf\u00d7h', values: { h: h, n_q_f: n_q_full, h_f: globalKvHeads, d_f: globalHd, d_vf: fullVHd }, resultValue: fullAttnPerLayer, bar: [{ type: 'attn', bytes: fullAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(fullAttnPerLayer) },
      { name: 'Attn_s', tip: 'Sliding attention layer: Q + K + V + O projections.', expr: 'h\u00d7(n_q_s\u00d7d_s) + h\u00d7(h_s\u00d7d_s) + h\u00d7(h_s\u00d7d_vs) + n_q_s\u00d7d_vs\u00d7h', values: { h: h, n_q_s: n_q_swa, h_s: slidingKvHeads, d_s: slidingHd, d_vs: slidingVHd }, resultValue: slidingAttnPerLayer, bar: [{ type: 'attn', bytes: slidingAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(slidingAttnPerLayer) },
    ];
    if (fullDenseCount + slidingDenseCount > 0) {
      formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer.', expr: ffnMats + '\u00d7h\u00d7I', values: { h: h, I: I }, resultValue: denseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(denseFfnPerLayer) });
    }
    if (nRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s\u00d7' + ffnMats + '\u00d7h\u00d7I_s', values: { N_s: nShared, h: h, I_s: Is }, resultValue: sharedPerLayer, bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * wtPrecB }], ibarVal: fmtWNum(sharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed experts FFN per MoE layer.', expr: 'N_e\u00d7' + ffnMats + '\u00d7h\u00d7I_m', values: { N_e: nRouted, h: h, I_m: Im }, resultValue: expertPerLayer, bar: [{ type: 'ffn-expert', bytes: expertPerLayer * wtPrecB }], ibarVal: fmtWNum(expertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: tieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: tieEmbed ? 'V\u00d7h' : '2\u00d7V\u00d7h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    patterns = [];
    if (fullDenseCount > 0) {
      var fDenseTotal = fullAttnPerLayer + denseFfnPerLayer;
      patterns.push({
        segs: [{ type: 'attn', ratio: fullAttnPerLayer / fDenseTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / fDenseTotal }],
        count: fullDenseCount,
        label: 'full attn + dense',
        bytes: fDenseTotal * wtPrecB
      });
    }
    if (fullMoeCount > 0) {
      var fMoeTotal = fullAttnPerLayer + sharedPerLayer + expertPerLayer;
      var fMoeSegs = [{ type: 'attn', ratio: fullAttnPerLayer / fMoeTotal }];
      if (nShared > 0) fMoeSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / fMoeTotal });
      fMoeSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / fMoeTotal });
      patterns.push({ segs: fMoeSegs, count: fullMoeCount, label: 'full attn + MoE', bytes: fMoeTotal * wtPrecB });
    }
    if (slidingDenseCount > 0) {
      var sDenseTotal = slidingAttnPerLayer + denseFfnPerLayer;
      patterns.push({
        segs: [{ type: 'attn', ratio: slidingAttnPerLayer / sDenseTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / sDenseTotal }],
        count: slidingDenseCount,
        label: 'sliding attn + dense',
        bytes: sDenseTotal * wtPrecB
      });
    }
    if (slidingMoeCount > 0) {
      var sMoeTotal = slidingAttnPerLayer + sharedPerLayer + expertPerLayer;
      var sMoeSegs = [{ type: 'attn', ratio: slidingAttnPerLayer / sMoeTotal }];
      if (nShared > 0) sMoeSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / sMoeTotal });
      sMoeSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / sMoeTotal });
      patterns.push({ segs: sMoeSegs, count: slidingMoeCount, label: 'sliding attn + MoE', bytes: sMoeTotal * wtPrecB });
    }
    legendTypes = nRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Full attention layers', value: fmtWNum(fullLayers) },
      { label: 'Sliding attention layers', value: fmtWNum(slidingLayers) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Full attn Q heads', value: fmtWNum(n_q_full) },
      { label: 'Full attn KV heads', value: fmtWNum(globalKvHeads) },
      { label: 'Global head dim', value: fmtWNum(globalHd) },
      { label: 'Full V head dim', value: fmtWNum(fullVHd) },
      { label: 'Full attn per layer', value: fmtWNum(fullAttnPerLayer) },
      { label: 'Sliding Q heads', value: fmtWNum(n_q_swa) },
      { label: 'Sliding KV heads', value: fmtWNum(slidingKvHeads) },
      { label: 'Sliding head dim', value: fmtWNum(slidingHd) },
      { label: 'Sliding V head dim', value: fmtWNum(slidingVHd) },
      { label: 'Sliding attn per layer', value: fmtWNum(slidingAttnPerLayer) },
    ];
    if (fullDenseCount + slidingDenseCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(fullDenseCount + slidingDenseCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(denseFfnPerLayer) });
    }
    if (fullMoeCount + slidingMoeCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(fullMoeCount + slidingMoeCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(nRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(nShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(Im) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(sharedPerLayer) });
      breakdown.push({ label: 'Routed experts per layer', value: fmtWNum(expertPerLayer) });
    }
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: tieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'qwen_linear_full_hybrid') {
    var fullLayers = f.full_attention_layers || 0;
    var linearLayers = f.linear_attention_layers || 0;

    var fullAttnPerLayer = h * (n_q * d_h) + h * (h_kv * d_h) + h * (h_kv * d_h) + (n_q * d_h) * h;

    var linKvHeads = f.linear_num_key_heads || 0;
    var linValHeads = f.linear_num_value_heads || 0;
    var linKeyHd = f.linear_key_head_dim || d_h;
    var linValHd = f.linear_value_head_dim || d_h;

    var linearAttnPerLayer = h * (n_q * d_h) + h * (linKvHeads * linKeyHd) + h * (linValHeads * linValHd) + (n_q * d_h) * h;

    var nRouted = wf.n_routed_experts || 0;
    var nShared = wf.n_shared_experts || 0;
    var Im = wf.moe_intermediate_size || 0;
    var Is = wf.shared_expert_intermediate_size || Im;

    var sharedPerLayer = nShared * ffnMats * h * Is;
    var expertPerLayer = nRouted * ffnMats * h * Im;

    var I = wf.intermediate_size || 0;
    var denseFfnPerLayer = ffnMats * h * I;

    var fullMoeCount = 0, fullDenseCount = 0, linearMoeCount = 0, linearDenseCount = 0;
    for (var i = 0; i < L; i++) {
      var isMoe = isMoeLayer(wf, i);
      if (i < fullLayers) {
        if (isMoe) fullMoeCount++; else fullDenseCount++;
      } else {
        if (isMoe) linearMoeCount++; else linearDenseCount++;
      }
    }

    attnParams = fullLayers * fullAttnPerLayer + linearLayers * linearAttnPerLayer;
    ffnDenseParams = (fullDenseCount + linearDenseCount) * denseFfnPerLayer;
    ffnSharedParams = (fullMoeCount + linearMoeCount) * sharedPerLayer;
    ffnExpertParams = (fullMoeCount + linearMoeCount) * expertPerLayer;

    var tieEmbed = wf.tie_word_embeddings;
    embedParams = tieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + ' linear + full attention hybrid';
    formulas = [
      { name: 'Attn_f', tip: 'Full attention layer: standard GQA Q + K + V + O.', expr: 'h\u00d7n_q\u00d7d_h + 2\u00d7h\u00d7h_kv\u00d7d_h + n_q\u00d7d_h\u00d7h', values: { h: h, n_q: n_q, h_kv: h_kv, d_h: d_h }, resultValue: fullAttnPerLayer, bar: [{ type: 'attn', bytes: fullAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(fullAttnPerLayer) },
      { name: 'Attn_l', tip: 'Linear attention layer: Q (shared) + K + V + O (shared).', expr: 'h\u00d7n_q\u00d7d_h + h\u00d7h_kl\u00d7d_kl + h\u00d7h_vl\u00d7d_vl + n_q\u00d7d_h\u00d7h', values: { h: h, n_q: n_q, h_kl: linKvHeads, d_kl: linKeyHd, h_vl: linValHeads, d_vl: linValHd, d_h: d_h }, resultValue: linearAttnPerLayer, bar: [{ type: 'attn', bytes: linearAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(linearAttnPerLayer) },
    ];
    if (fullDenseCount + linearDenseCount > 0) {
      formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer.', expr: ffnMats + '\u00d7h\u00d7I', values: { h: h, I: I }, resultValue: denseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(denseFfnPerLayer) });
    }
    if (nRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s\u00d7' + ffnMats + '\u00d7h\u00d7I_s', values: { N_s: nShared, h: h, I_s: Is }, resultValue: sharedPerLayer, bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * wtPrecB }], ibarVal: fmtWNum(sharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed experts FFN per MoE layer.', expr: 'N_e\u00d7' + ffnMats + '\u00d7h\u00d7I_m', values: { N_e: nRouted, h: h, I_m: Im }, resultValue: expertPerLayer, bar: [{ type: 'ffn-expert', bytes: expertPerLayer * wtPrecB }], ibarVal: fmtWNum(expertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: tieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: tieEmbed ? 'V\u00d7h' : '2\u00d7V\u00d7h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    patterns = [];
    if (fullDenseCount > 0) {
      var fDenseTotal = fullAttnPerLayer + denseFfnPerLayer;
      patterns.push({
        segs: [{ type: 'attn', ratio: fullAttnPerLayer / fDenseTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / fDenseTotal }],
        count: fullDenseCount,
        label: 'full attn + dense',
        bytes: fDenseTotal * wtPrecB
      });
    }
    if (fullMoeCount > 0) {
      var fMoeTotal = fullAttnPerLayer + sharedPerLayer + expertPerLayer;
      var fMoeSegs = [{ type: 'attn', ratio: fullAttnPerLayer / fMoeTotal }];
      if (nShared > 0) fMoeSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / fMoeTotal });
      fMoeSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / fMoeTotal });
      patterns.push({ segs: fMoeSegs, count: fullMoeCount, label: 'full attn + MoE', bytes: fMoeTotal * wtPrecB });
    }
    if (linearDenseCount > 0) {
      var lDenseTotal = linearAttnPerLayer + denseFfnPerLayer;
      patterns.push({
        segs: [{ type: 'attn', ratio: linearAttnPerLayer / lDenseTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / lDenseTotal }],
        count: linearDenseCount,
        label: 'linear attn + dense',
        bytes: lDenseTotal * wtPrecB
      });
    }
    if (linearMoeCount > 0) {
      var lMoeTotal = linearAttnPerLayer + sharedPerLayer + expertPerLayer;
      var lMoeSegs = [{ type: 'attn', ratio: linearAttnPerLayer / lMoeTotal }];
      if (nShared > 0) lMoeSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / lMoeTotal });
      lMoeSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / lMoeTotal });
      patterns.push({ segs: lMoeSegs, count: linearMoeCount, label: 'linear attn + MoE', bytes: lMoeTotal * wtPrecB });
    }
    legendTypes = nRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Full attention layers', value: fmtWNum(fullLayers) },
      { label: 'Linear attention layers', value: fmtWNum(linearLayers) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads (Q)', value: fmtWNum(n_q) },
      { label: 'Full KV heads', value: fmtWNum(h_kv) },
      { label: 'Head dim', value: fmtWNum(d_h) },
      { label: 'Full attn per layer', value: fmtWNum(fullAttnPerLayer) },
      { label: 'Linear key heads', value: fmtWNum(linKvHeads) },
      { label: 'Linear value heads', value: fmtWNum(linValHeads) },
      { label: 'Linear key head dim', value: fmtWNum(linKeyHd) },
      { label: 'Linear value head dim', value: fmtWNum(linValHd) },
      { label: 'Linear attn per layer', value: fmtWNum(linearAttnPerLayer) },
    ];
    if (fullDenseCount + linearDenseCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(fullDenseCount + linearDenseCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(denseFfnPerLayer) });
    }
    if (fullMoeCount + linearMoeCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(fullMoeCount + linearMoeCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(nRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(nShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(Im) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(sharedPerLayer) });
      breakdown.push({ label: 'Routed experts per layer', value: fmtWNum(expertPerLayer) });
    }
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: tieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });


  } else if (formula === 'glm5_next_hybrid') {
    // GLM-5.3-Flash alternates KDA linear layers with sparse MLA layers.
    var glmLayerTypes = Array.isArray(f.layer_types) ? f.layer_types : [];
    var glmConfiguredSparseIds = Array.isArray(f.sparse_attention_layer_ids) ? f.sparse_attention_layer_ids : [];
    var glmSparseIds = glmConfiguredSparseIds.length > 0
      ? glmConfiguredSparseIds
      : (glmLayerTypes.length > 0
        ? glmLayerTypes.map(function (type, index) { return type === 'deepseek_sparse_attention' ? index : -1; }).filter(function (index) { return index >= 0; })
        : []);
    var glmSparseLayers = glmSparseIds.length || f.sparse_attention_layers || 0;
    var glmLinearLayers = f.linear_attention_layers || (L - glmSparseLayers);

    var glmQLoraRank = wf.q_lora_rank || f.q_lora_rank || 0;
    var glmQkRopeHd = f.qk_rope_head_dim || 0;
    var glmQkNopeHd = f.qk_nope_head_dim || 0;
    var glmKvLoraRank = f.kv_lora_rank || 0;
    var glmQkHd = f.qk_head_dim || (glmQkNopeHd + glmQkRopeHd);
    var glmVHeadDim = wf.v_head_dim || f.v_head_dim || glmQkHd;
    var glmWqa = h * glmQLoraRank;
    var glmWqb = glmQLoraRank * (n_q * glmQkHd);
    var glmWkva = h * (glmKvLoraRank + glmQkRopeHd);
    var glmWkvb = glmKvLoraRank * n_q * (glmQkNopeHd + glmVHeadDim);
    var glmWo = (n_q * glmVHeadDim) * h;
    var glmSparseAttnPerLayer = glmWqa + glmWqb + glmWkva + glmWkvb + glmWo;

    var glmIdxHd = f.index_head_dim || 0;
    var glmIdxHeads = f.index_n_heads || 0;
    var glmIndexPool = f.index_kpool || 1;
    var glmIdxPerLayer = (h * glmIdxHd) + (glmQLoraRank * glmIdxHeads * glmIdxHd) + (h * glmIdxHeads) + (glmIdxHd * glmIndexPool);

    // KDA projections and state-control weights. dt/A and normalization
    // vectors are small relative to these matrices and are intentionally not
    // expanded into separate bars.
    var glmLinearHeadCount = f.linear_num_heads || 0;
    var glmLinearHeadDim = f.linear_head_dim || 0;
    var glmLinearQkvDim = glmLinearHeadCount * glmLinearHeadDim;
    var glmLinearAttnPerLayer = (3 * h * glmLinearQkvDim) + (3 * glmLinearQkvDim * (f.linear_conv_kernel_dim || 0)) + (h * glmLinearHeadDim) + (glmLinearHeadDim * glmLinearQkvDim) + (h * glmLinearHeadCount) + (h * glmLinearHeadDim) + (glmLinearHeadDim * glmLinearQkvDim) + (glmLinearQkvDim * h);

    var glmNRouted = wf.n_routed_experts || 0;
    var glmNShared = wf.n_shared_experts || 0;
    var glmIm = wf.moe_intermediate_size || 0;
    var glmIs = wf.shared_expert_intermediate_size || glmIm;
    var glmI = wf.intermediate_size || 0;
    var glmDenseFfnPerLayer = ffnMats * h * glmI;
    var glmSharedPerLayer = glmNShared * ffnMats * h * glmIs;
    var glmExpertPerLayer = glmNRouted * ffnMats * h * glmIm;
    var glmDenseCount = 0, glmMoeCount = 0;
    for (var glmLayerIndex = 0; glmLayerIndex < L; glmLayerIndex++) {
      if (isMoeLayer(wf, glmLayerIndex)) glmMoeCount++; else glmDenseCount++;
    }

    attnParams = glmSparseLayers * (glmSparseAttnPerLayer + glmIdxPerLayer) + glmLinearLayers * glmLinearAttnPerLayer;
    ffnDenseParams = glmDenseCount * glmDenseFfnPerLayer;
    ffnSharedParams = glmMoeCount * glmSharedPerLayer;
    ffnExpertParams = glmMoeCount * glmExpertPerLayer;
    var glmTieEmbed = wf.tie_word_embeddings;
    embedParams = glmTieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + ' KDA linear + sparse MLA';
    formulas = [
      { name: 'Attn_s', tip: 'Sparse MLA attention per sparse layer: Q/KV LoRA + output projection.', expr: 'h×q_r + q_r×n_q×qk + h×(d_c+d_r) + d_c×n_q×(qk_nope+d_v) + n_q×d_v×h', values: { h: h, q_r: glmQLoraRank, n_q: n_q, qk: glmQkHd, d_c: glmKvLoraRank, d_r: glmQkRopeHd, qk_nope: glmQkNopeHd, d_v: glmVHeadDim }, resultValue: glmSparseAttnPerLayer, bar: [{ type: 'attn', bytes: glmSparseAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(glmSparseAttnPerLayer) },
      { name: 'Idx', tip: 'Indexer weights on sparse MLA layers only.', expr: 'h×d_idx + q_r×h_idx×d_idx + h×h_idx + d_idx×k_pool', values: { h: h, q_r: glmQLoraRank, h_idx: glmIdxHeads, d_idx: glmIdxHd, k_pool: glmIndexPool }, resultValue: glmIdxPerLayer, bar: [{ type: 'attn', bytes: glmIdxPerLayer * wtPrecB }], ibarVal: fmtWNum(glmIdxPerLayer) },
      { name: 'Attn_l', tip: 'KDA linear attention per layer, including Q/K/V, short convolution, gates and output projection.', expr: '3×h×qkv + 3×qkv×k_c + h×d_l + d_l×qkv + h×h_l + h×d_l + d_l×qkv + qkv×h', values: { h: h, qkv: glmLinearQkvDim, k_c: f.linear_conv_kernel_dim || 0, d_l: glmLinearHeadDim, h_l: glmLinearHeadCount }, resultValue: glmLinearAttnPerLayer, bar: [{ type: 'attn', bytes: glmLinearAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(glmLinearAttnPerLayer) },
    ];
    if (glmDenseCount > 0) formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer.', expr: ffnMats + '×h×I', values: { h: h, I: glmI }, resultValue: glmDenseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: glmDenseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(glmDenseFfnPerLayer) });
    if (glmNRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s×' + ffnMats + '×h×I_s', values: { N_s: glmNShared, h: h, I_s: glmIs }, resultValue: glmSharedPerLayer, bar: [{ type: 'ffn-shared', bytes: glmSharedPerLayer * wtPrecB }], ibarVal: fmtWNum(glmSharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed expert FFN per MoE layer.', expr: 'N_e×' + ffnMats + '×h×I_m', values: { N_e: glmNRouted, h: h, I_m: glmIm }, resultValue: glmExpertPerLayer, bar: [{ type: 'ffn-expert', bytes: glmExpertPerLayer * wtPrecB }], ibarVal: fmtWNum(glmExpertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: glmTieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: glmTieEmbed ? 'V×h' : '2×V×h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    var glmSparseTotal = glmSparseAttnPerLayer + glmIdxPerLayer + (glmMoeCount > 0 ? glmSharedPerLayer + glmExpertPerLayer : glmDenseFfnPerLayer);
    var glmLinearTotal = glmLinearAttnPerLayer + (glmMoeCount > 0 ? glmSharedPerLayer + glmExpertPerLayer : glmDenseFfnPerLayer);
    var glmSparseSegs = [{ type: 'attn', ratio: (glmSparseAttnPerLayer + glmIdxPerLayer) / glmSparseTotal }];
    var glmLinearSegs = [{ type: 'attn', ratio: glmLinearAttnPerLayer / glmLinearTotal }];
    if (glmMoeCount > 0) {
      if (glmNShared > 0) { glmSparseSegs.push({ type: 'ffn-shared', ratio: glmSharedPerLayer / glmSparseTotal }); glmLinearSegs.push({ type: 'ffn-shared', ratio: glmSharedPerLayer / glmLinearTotal }); }
      glmSparseSegs.push({ type: 'ffn-expert', ratio: glmExpertPerLayer / glmSparseTotal });
      glmLinearSegs.push({ type: 'ffn-expert', ratio: glmExpertPerLayer / glmLinearTotal });
    } else {
      glmSparseSegs.push({ type: 'ffn-dense', ratio: glmDenseFfnPerLayer / glmSparseTotal });
      glmLinearSegs.push({ type: 'ffn-dense', ratio: glmDenseFfnPerLayer / glmLinearTotal });
    }
    patterns = [
      { segs: glmSparseSegs, count: glmSparseLayers, label: 'sparse MLA + indexer', bytes: glmSparseTotal * wtPrecB },
      { segs: glmLinearSegs, count: glmLinearLayers, label: 'KDA linear', bytes: glmLinearTotal * wtPrecB },
    ];
    legendTypes = glmNRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Sparse MLA layers', value: fmtWNum(glmSparseLayers) },
      { label: 'KDA linear layers', value: fmtWNum(glmLinearLayers) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads', value: fmtWNum(n_q) },
      { label: 'KV LoRA rank', value: fmtWNum(glmKvLoraRank) },
      { label: 'Sparse attention per layer', value: fmtWNum(glmSparseAttnPerLayer) },
      { label: 'Indexer params per sparse layer', value: fmtWNum(glmIdxPerLayer) },
      { label: 'KDA attention per layer', value: fmtWNum(glmLinearAttnPerLayer) },
    ];
    if (glmDenseCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(glmDenseCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(glmDenseFfnPerLayer) });
    }
    if (glmMoeCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(glmMoeCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(glmNRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(glmNShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(glmIm) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(glmSharedPerLayer) });
      breakdown.push({ label: 'Routed expert per layer', value: fmtWNum(glmExpertPerLayer) });
    }
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: glmTieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'qwen_qsa_gdn_hybrid') {
    // Qwen3.8-Flash-Next has raw-GQA QSA layers and GDN linear layers. The
    // auxiliary n-gram/PLE table is off-accelerator and is shown separately.
    var qsaFullLayers = f.full_attention_layers || 0;
    var qsaLinearLayers = f.linear_attention_layers || (L - qsaFullLayers);
    var qsaFullAttnPerLayer = h * (n_q * d_h) + h * (h_kv * d_h) + h * (h_kv * d_h) + (n_q * d_h) * h;
    var qsaLinKeyHeads = f.linear_num_key_heads || 0;
    var qsaLinValueHeads = f.linear_num_value_heads || 0;
    var qsaLinKeyHd = f.linear_key_head_dim || d_h;
    var qsaLinValueHd = f.linear_value_head_dim || d_h;
    var qsaLinearAttnPerLayer = h * (n_q * d_h) + h * (qsaLinKeyHeads * qsaLinKeyHd) + h * (qsaLinValueHeads * qsaLinValueHd) + (n_q * d_h) * h;
    var qsaIdxHd = f.indexer_head_dim || 0;
    var qsaIdxKvHeads = f.indexer_kv_heads || 1;
    var qsaIdxHeads = f.indexer_n_heads || 0;
    var qsaIdxPerFullLayer = (h * qsaIdxHd * qsaIdxKvHeads) + (h * qsaIdxHeads);

    var qsaNRouted = wf.n_routed_experts || 0;
    var qsaNShared = wf.n_shared_experts || 0;
    var qsaIm = wf.moe_intermediate_size || 0;
    var qsaIs = wf.shared_expert_intermediate_size || qsaIm;
    var qsaI = wf.intermediate_size || 0;
    var qsaDenseFfnPerLayer = ffnMats * h * qsaI;
    var qsaSharedPerLayer = qsaNShared * ffnMats * h * qsaIs;
    var qsaExpertPerLayer = qsaNRouted * ffnMats * h * qsaIm;
    var qsaDenseCount = 0, qsaMoeCount = 0;
    for (var qsaLayerIndex = 0; qsaLayerIndex < L; qsaLayerIndex++) {
      if (isMoeLayer(wf, qsaLayerIndex)) qsaMoeCount++; else qsaDenseCount++;
    }

    attnParams = qsaFullLayers * (qsaFullAttnPerLayer + qsaIdxPerFullLayer) + qsaLinearLayers * qsaLinearAttnPerLayer;
    ffnDenseParams = qsaDenseCount * qsaDenseFfnPerLayer;
    ffnSharedParams = qsaMoeCount * qsaSharedPerLayer;
    ffnExpertParams = qsaMoeCount * qsaExpertPerLayer;
    var qsaTieEmbed = wf.tie_word_embeddings;
    embedParams = qsaTieEmbed ? (V * h) : (2 * V * h);

    var qsaNgramParams = (f.ngram_vocab_size_base || wf.ngram_vocab_size_base || 0) * (f.ple_embed_dim || wf.ple_embed_dim || 0);
    formulaTitle = model.label + ' GDN linear + QSA';
    formulas = [
      { name: 'Attn_f', tip: 'QSA full-attention layer: raw GQA Q + K + V + O projections.', expr: 'h×n_q×d_h + 2×h×h_kv×d_h + n_q×d_h×h', values: { h: h, n_q: n_q, h_kv: h_kv, d_h: d_h }, resultValue: qsaFullAttnPerLayer, bar: [{ type: 'attn', bytes: qsaFullAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(qsaFullAttnPerLayer) },
      { name: 'Idx', tip: 'QSA indexer projection per full-attention layer; indexer K cache remains MQA.', expr: 'h×h_idx_kv×d_idx + h×h_idx', values: { h: h, h_idx_kv: qsaIdxKvHeads, d_idx: qsaIdxHd, h_idx: qsaIdxHeads }, resultValue: qsaIdxPerFullLayer, bar: [{ type: 'attn', bytes: qsaIdxPerFullLayer * wtPrecB }], ibarVal: fmtWNum(qsaIdxPerFullLayer) },
      { name: 'Attn_l', tip: 'GDN linear layer: shared Q plus linear K/V and output projections.', expr: 'h×n_q×d_h + h×h_kl×d_kl + h×h_vl×d_vl + n_q×d_h×h', values: { h: h, n_q: n_q, d_h: d_h, h_kl: qsaLinKeyHeads, d_kl: qsaLinKeyHd, h_vl: qsaLinValueHeads, d_vl: qsaLinValueHd }, resultValue: qsaLinearAttnPerLayer, bar: [{ type: 'attn', bytes: qsaLinearAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(qsaLinearAttnPerLayer) },
    ];
    if (qsaDenseCount > 0) formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer.', expr: ffnMats + '×h×I', values: { h: h, I: qsaI }, resultValue: qsaDenseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: qsaDenseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(qsaDenseFfnPerLayer) });
    if (qsaNRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s×' + ffnMats + '×h×I_s', values: { N_s: qsaNShared, h: h, I_s: qsaIs }, resultValue: qsaSharedPerLayer, bar: [{ type: 'ffn-shared', bytes: qsaSharedPerLayer * wtPrecB }], ibarVal: fmtWNum(qsaSharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed expert FFN per MoE layer.', expr: 'N_e×' + ffnMats + '×h×I_m', values: { N_e: qsaNRouted, h: h, I_m: qsaIm }, resultValue: qsaExpertPerLayer, bar: [{ type: 'ffn-expert', bytes: qsaExpertPerLayer * wtPrecB }], ibarVal: fmtWNum(qsaExpertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: qsaTieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: qsaTieEmbed ? 'V×h' : '2×V×h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    var qsaFullTotal = qsaFullAttnPerLayer + qsaIdxPerFullLayer + (qsaMoeCount > 0 ? qsaSharedPerLayer + qsaExpertPerLayer : qsaDenseFfnPerLayer);
    var qsaLinearTotal = qsaLinearAttnPerLayer + (qsaMoeCount > 0 ? qsaSharedPerLayer + qsaExpertPerLayer : qsaDenseFfnPerLayer);
    var qsaFullSegs = [{ type: 'attn', ratio: (qsaFullAttnPerLayer + qsaIdxPerFullLayer) / qsaFullTotal }];
    var qsaLinearSegs = [{ type: 'attn', ratio: qsaLinearAttnPerLayer / qsaLinearTotal }];
    if (qsaMoeCount > 0) {
      if (qsaNShared > 0) { qsaFullSegs.push({ type: 'ffn-shared', ratio: qsaSharedPerLayer / qsaFullTotal }); qsaLinearSegs.push({ type: 'ffn-shared', ratio: qsaSharedPerLayer / qsaLinearTotal }); }
      qsaFullSegs.push({ type: 'ffn-expert', ratio: qsaExpertPerLayer / qsaFullTotal });
      qsaLinearSegs.push({ type: 'ffn-expert', ratio: qsaExpertPerLayer / qsaLinearTotal });
    } else {
      qsaFullSegs.push({ type: 'ffn-dense', ratio: qsaDenseFfnPerLayer / qsaFullTotal });
      qsaLinearSegs.push({ type: 'ffn-dense', ratio: qsaDenseFfnPerLayer / qsaLinearTotal });
    }
    patterns = [
      { segs: qsaFullSegs, count: qsaFullLayers, label: 'QSA full + indexer', bytes: qsaFullTotal * wtPrecB },
      { segs: qsaLinearSegs, count: qsaLinearLayers, label: 'GDN linear', bytes: qsaLinearTotal * wtPrecB },
    ];
    legendTypes = qsaNRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'QSA full layers', value: fmtWNum(qsaFullLayers) },
      { label: 'GDN linear layers', value: fmtWNum(qsaLinearLayers) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Full attention heads', value: fmtWNum(n_q) },
      { label: 'Full KV heads', value: fmtWNum(h_kv) },
      { label: 'Full head dim', value: fmtWNum(d_h) },
      { label: 'Full attn per layer', value: fmtWNum(qsaFullAttnPerLayer) },
      { label: 'Indexer params per full layer', value: fmtWNum(qsaIdxPerFullLayer) },
      { label: 'Linear attn per layer', value: fmtWNum(qsaLinearAttnPerLayer) },
    ];
    if (qsaDenseCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(qsaDenseCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(qsaDenseFfnPerLayer) });
    }
    if (qsaMoeCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(qsaMoeCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(qsaNRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(qsaNShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(qsaIm) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(qsaSharedPerLayer) });
      breakdown.push({ label: 'Routed expert per layer', value: fmtWNum(qsaExpertPerLayer) });
    }
    if (qsaNgramParams > 0) breakdown.push({ label: 'N-gram / PLE params (external)', value: fmtWNum(qsaNgramParams), tip: 'Auxiliary n-gram table; excluded from accelerator-resident transformer weight total and from KV cache.' });
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: qsaTieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'kda_gated_mla') {
    var fullLayers = f.full_attention_layers || 0;
    var linearLayers = f.linear_attention_layers || 0;

    // Gated MLA full-attention layers (Kimi K3 Gated MLA: Q/KV LoRA + output gate)
    var qLoraRank = wf.q_lora_rank || 0;
    var qkRopeHd = f.qk_rope_head_dim || 0;
    var qkNopeHd = f.qk_nope_head_dim || 0;
    var kvLoraRank = f.kv_lora_rank || 0;
    var qkHd = f.qk_head_dim || (qkNopeHd + qkRopeHd);
    var Wqa = h * qLoraRank;
    var Wqb = qLoraRank * (n_q * qkHd);
    var Wkva = h * (kvLoraRank + qkRopeHd);
    var Wkvb = kvLoraRank * n_q * (qkNopeHd + d_v);
    var Wg_mla = h * (n_q * d_v);
    var Wo_mla = (n_q * d_v) * h;
    var mlaAttnPerLayer = Wqa + Wqb + Wkva + Wkvb + Wg_mla + Wo_mla;

    // KDA linear-attention layers (Q/K/V + delta-rule gate + output)
    var linKvHeads = f.linear_num_key_heads || 0;
    var linValHeads = f.linear_num_value_heads || 0;
    var linKeyHd = f.linear_key_head_dim || d_h;
    var linValHd = f.linear_value_head_dim || d_h;
    var kdaProj = linValHeads * linValHd;
    var Wq_kda = h * kdaProj;
    var Wk_kda = h * (linKvHeads * linKeyHd);
    var Wv_kda = h * kdaProj;
    var Wfa = h * linKeyHd;
    var Wfb = linKeyHd * kdaProj;
    var Wb = h * linKvHeads;
    var Wg_kda = h * kdaProj;
    var Wo_kda = kdaProj * h;
    var kdaAttnPerLayer = Wq_kda + Wk_kda + Wv_kda + Wfa + Wfb + Wb + Wg_kda + Wo_kda;

    var I = wf.intermediate_size || 0;
    var denseFfnPerLayer = ffnMats * h * I;

    var nRouted = wf.n_routed_experts || 0;
    var nShared = wf.n_shared_experts || 0;
    var Im = wf.moe_intermediate_size || I;
    var Is = wf.shared_expert_intermediate_size || Im;
    var latent = wf.routed_expert_hidden_size || h;

    // Stable LatentMoE: experts run at the latent dimension, plus shared latent down/up projections
    var sharedPerLayer = nShared * ffnMats * h * Is;
    var expertPerLayer = nRouted * ffnMats * latent * Im;
    if (latent !== h) {
      expertPerLayer += 2 * h * latent;
    }

    var denseFfnCount = 0, moeFfnCount = 0;
    for (var i = 0; i < L; i++) {
      if (isMoeLayer(wf, i)) moeFfnCount++; else denseFfnCount++;
    }

    attnParams = fullLayers * mlaAttnPerLayer + linearLayers * kdaAttnPerLayer;
    ffnDenseParams = denseFfnCount * denseFfnPerLayer;
    ffnSharedParams = moeFfnCount * sharedPerLayer;
    ffnExpertParams = moeFfnCount * expertPerLayer;

    var tieEmbed = wf.tie_word_embeddings;
    embedParams = tieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + ' KDA + Gated MLA';
    formulas = [
      { name: 'Attn_f', tip: 'Gated MLA attention per layer: Q LoRA down/up + KV LoRA down/up + output gate + O projection.', expr: 'h\u00d7q_r + q_r\u00d7n_q\u00d7qk + h\u00d7(d_c+d_r) + d_c\u00d7n_q\u00d7(qk_nope+d_v) + n_q\u00d7d_v\u00d7h + n_q\u00d7d_v\u00d7h', values: { h: h, q_r: qLoraRank, n_q: n_q, qk: qkHd, d_c: kvLoraRank, d_r: qkRopeHd, qk_nope: qkNopeHd, d_v: d_v }, resultValue: mlaAttnPerLayer, bar: [{ type: 'attn', bytes: mlaAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(mlaAttnPerLayer) },
      { name: 'Attn_l', tip: 'KDA linear attention per layer: Q + K + V + delta gate (f_a/f_b) + beta + full-rank gate + O.', expr: 'h\u00d7n_q\u00d7d_kl + h\u00d7h_kl\u00d7d_kl + h\u00d7h_vl\u00d7d_vl + h\u00d7d_kl + d_kl\u00d7h_vl\u00d7d_vl + h\u00d7h_kl + h\u00d7h_vl\u00d7d_vl + h_vl\u00d7d_vl\u00d7h', values: { h: h, n_q: n_q, h_kl: linKvHeads, d_kl: linKeyHd, h_vl: linValHeads, d_vl: linValHd }, resultValue: kdaAttnPerLayer, bar: [{ type: 'attn', bytes: kdaAttnPerLayer * wtPrecB }], ibarVal: fmtWNum(kdaAttnPerLayer) },
    ];
    if (denseFfnCount > 0) {
      formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer (SiTU-GLU, 3 matrices).', expr: ffnMats + '\u00d7h\u00d7I', values: { h: h, I: I }, resultValue: denseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(denseFfnPerLayer) });
    }
    if (nRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer (2 experts fused).', expr: 'N_s\u00d7' + ffnMats + '\u00d7h\u00d7I_s', values: { N_s: nShared, h: h, I_s: Is }, resultValue: sharedPerLayer, bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * wtPrecB }], ibarVal: fmtWNum(sharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed experts per MoE layer at latent dim (incl. latent down/up projections).', expr: 'N_e\u00d7' + ffnMats + '\u00d7h_lat\u00d7I_m + 2\u00d7h\u00d7h_lat', values: { N_e: nRouted, h_lat: latent, h: h, I_m: Im }, resultValue: expertPerLayer, bar: [{ type: 'ffn-expert', bytes: expertPerLayer * wtPrecB }], ibarVal: fmtWNum(expertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: tieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: tieEmbed ? 'V\u00d7h' : '2\u00d7V\u00d7h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    patterns = [];
    // Classify each layer as KDA or Gated MLA using the official layer list.
    var kdaLayerSet = {};
    for (var ki = 0; ki < L; ki++) kdaLayerSet[ki] = false;
    if (Array.isArray(f.kda_layers)) {
      f.kda_layers.forEach(function (idx) { kdaLayerSet[idx - 1] = true; });
    } else {
      for (var ki2 = 0; ki2 < linearLayers; ki2++) kdaLayerSet[ki2] = true;
    }
    var fullDenseCount = 0, fullMoeCount = 0, kdaDenseCount = 0, kdaMoeCount = 0;
    for (var li = 0; li < L; li++) {
      var isMoeLi = isMoeLayer(wf, li);
      if (kdaLayerSet[li]) { if (isMoeLi) kdaMoeCount++; else kdaDenseCount++; }
      else { if (isMoeLi) fullMoeCount++; else fullDenseCount++; }
    }
    if (kdaDenseCount > 0) {
      var kdTotal = kdaAttnPerLayer + denseFfnPerLayer;
      patterns.push({ segs: [{ type: 'attn', ratio: kdaAttnPerLayer / kdTotal }, { type: 'ffn-dense', ratio: denseFfnPerLayer / kdTotal }], count: kdaDenseCount, label: 'KDA + dense', bytes: kdTotal * wtPrecB });
    }
    if (fullMoeCount > 0) {
      var fMTotal = mlaAttnPerLayer + sharedPerLayer + expertPerLayer;
      var fMSegs = [{ type: 'attn', ratio: mlaAttnPerLayer / fMTotal }];
      if (nShared > 0) fMSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / fMTotal });
      fMSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / fMTotal });
      patterns.push({ segs: fMSegs, count: fullMoeCount, label: 'gated MLA + MoE', bytes: fMTotal * wtPrecB });
    }
    if (kdaMoeCount > 0) {
      var kMTotal = kdaAttnPerLayer + sharedPerLayer + expertPerLayer;
      var kMSegs = [{ type: 'attn', ratio: kdaAttnPerLayer / kMTotal }];
      if (nShared > 0) kMSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / kMTotal });
      kMSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / kMTotal });
      patterns.push({ segs: kMSegs, count: kdaMoeCount, label: 'KDA + MoE', bytes: kMTotal * wtPrecB });
    }
    legendTypes = nRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Gated MLA layers', value: fmtWNum(fullLayers) },
      { label: 'KDA linear layers', value: fmtWNum(linearLayers) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads', value: fmtWNum(n_q) },
      { label: 'KDA head dim', value: fmtWNum(linKeyHd) },
      { label: 'MLA attn per layer', value: fmtWNum(mlaAttnPerLayer) },
      { label: 'KDA attn per layer', value: fmtWNum(kdaAttnPerLayer) },
    ];
    if (denseFfnCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(denseFfnCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(denseFfnPerLayer) });
    }
    if (moeFfnCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(moeFfnCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(nRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(nShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(Im) });
      breakdown.push({ label: 'Routed expert hidden size', value: fmtWNum(latent) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(sharedPerLayer) });
      breakdown.push({ label: 'Routed experts per layer', value: fmtWNum(expertPerLayer) });
    }
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: tieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else if (formula === 'msa_gqa') {
    var Wq = h * (n_q * d_h);
    var Wk = h * (h_kv * d_h);
    var Wv = h * (h_kv * d_v);
    var Wo = (n_q * d_v) * h;
    var attnPerLayer = Wq + Wk + Wv + Wo;

    var idxHd = f.sparse_index_dim || f.index_head_dim || 0;
    var idxHeads = f.sparse_num_index_heads || f.index_n_heads || 0;
    var sparseFreq = f.sparse_attention_freq;
    var sparseLayerCount = sparseFreq ? sparseFreq.filter(function(v) { return v === 1; }).length : 0;
    var idxPerSparseLayer = h * (idxHeads * idxHd);

    var I = wf.intermediate_size || 0;
    var denseFfnPerLayer = ffnMats * h * I;

    var nRouted = wf.n_routed_experts || 0;
    var nShared = wf.n_shared_experts || 0;
    var Im = wf.moe_intermediate_size || I;
    var Is = wf.shared_expert_intermediate_size || Im;

    var sharedPerLayer = nShared * ffnMats * h * Is;
    var expertPerLayer = nRouted * ffnMats * h * Im;

    var fullAttnLayerCount = L - sparseLayerCount;
    var sparseAttnLayerCount = sparseLayerCount;

    var denseFfnCount = 0, moeFfnCount = 0;
    for (var i = 0; i < L; i++) {
      if (isMoeLayer(wf, i)) moeFfnCount++; else denseFfnCount++;
    }

    attnParams = L * attnPerLayer + sparseLayerCount * idxPerSparseLayer;
    ffnDenseParams = denseFfnCount * denseFfnPerLayer;
    ffnSharedParams = moeFfnCount * sharedPerLayer;
    ffnExpertParams = moeFfnCount * expertPerLayer;

    var tieEmbed = wf.tie_word_embeddings;
    embedParams = tieEmbed ? (V * h) : (2 * V * h);

    formulaTitle = model.label + ' MSA + GQA';
    formulas = [
      { name: 'Attn', tip: 'Standard GQA attention per layer: Q + K + V + O.', expr: 'h\u00d7n_q\u00d7d_h + 2\u00d7h\u00d7h_kv\u00d7d_h + n_q\u00d7d_h\u00d7h', values: { h: h, n_q: n_q, h_kv: h_kv, d_h: d_h, d_v: d_v }, resultValue: attnPerLayer, bar: [{ type: 'attn', bytes: attnPerLayer * wtPrecB }], ibarVal: fmtWNum(attnPerLayer) },
    ];
    if (sparseLayerCount > 0 && idxPerSparseLayer > 0) {
      formulas.push({ name: 'Idx', tip: 'Sparse index key projection per sparse layer.', expr: 'h\u00d7h_idx\u00d7d_idx', values: { h: h, h_idx: idxHeads, d_idx: idxHd }, resultValue: idxPerSparseLayer, bar: [{ type: 'attn', bytes: idxPerSparseLayer * wtPrecB }], ibarVal: fmtWNum(idxPerSparseLayer) });
    }
    if (denseFfnCount > 0) {
      formulas.push({ name: 'FFN_d', tip: 'Dense FFN per layer.', expr: ffnMats + '\u00d7h\u00d7I', values: { h: h, I: I }, resultValue: denseFfnPerLayer, bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * wtPrecB }], ibarVal: fmtWNum(denseFfnPerLayer) });
    }
    if (nRouted > 0) {
      formulas.push(
        { name: 'FFN_s', tip: 'Shared expert FFN per MoE layer.', expr: 'N_s\u00d7' + ffnMats + '\u00d7h\u00d7I_s', values: { N_s: nShared, h: h, I_s: Is }, resultValue: sharedPerLayer, bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * wtPrecB }], ibarVal: fmtWNum(sharedPerLayer) },
        { name: 'FFN_e', tip: 'Routed experts FFN per MoE layer.', expr: 'N_e\u00d7' + ffnMats + '\u00d7h\u00d7I_m', values: { N_e: nRouted, h: h, I_m: Im }, resultValue: expertPerLayer, bar: [{ type: 'ffn-expert', bytes: expertPerLayer * wtPrecB }], ibarVal: fmtWNum(expertPerLayer) }
      );
    }
    formulas.push({ name: 'Embed', tip: tieEmbed ? 'Embedding only (tied with lm_head).' : 'Embedding + lm_head (untied).', expr: tieEmbed ? 'V\u00d7h' : '2\u00d7V\u00d7h', values: { V: V, h: h }, resultValue: embedParams, bar: [{ type: 'embed', bytes: embedParams * wtPrecB }], ibarVal: fmtWNum(embedParams) });

    patterns = [];
    if (fullAttnLayerCount > 0) {
      var fTotal = attnPerLayer + (denseFfnCount > 0 ? denseFfnPerLayer : sharedPerLayer + expertPerLayer);
      var fSegs = [{ type: 'attn', ratio: attnPerLayer / fTotal }];
      if (denseFfnCount > 0) { fSegs.push({ type: 'ffn-dense', ratio: denseFfnPerLayer / fTotal }); }
      else { if (nShared > 0) fSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / fTotal }); fSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / fTotal }); }
      patterns.push({ segs: fSegs, count: fullAttnLayerCount, label: 'full attn', bytes: fTotal * wtPrecB });
    }
    if (sparseAttnLayerCount > 0) {
      var sTotal = attnPerLayer + idxPerSparseLayer + (moeFfnCount > 0 ? sharedPerLayer + expertPerLayer : denseFfnPerLayer);
      var sSegs = [{ type: 'attn', ratio: (attnPerLayer + idxPerSparseLayer) / sTotal }];
      if (moeFfnCount > 0) {
        if (nShared > 0) sSegs.push({ type: 'ffn-shared', ratio: sharedPerLayer / sTotal });
        sSegs.push({ type: 'ffn-expert', ratio: expertPerLayer / sTotal });
      } else {
        sSegs.push({ type: 'ffn-dense', ratio: denseFfnPerLayer / sTotal });
      }
      patterns.push({ segs: sSegs, count: sparseAttnLayerCount, label: 'sparse attn', bytes: sTotal * wtPrecB });
    }
    legendTypes = nRouted > 0 ? ['attn', 'ffn-dense', 'ffn-shared', 'ffn-expert', 'embed'] : ['attn', 'ffn-dense', 'embed'];

    breakdown = [
      { label: 'Layers', value: fmtWNum(L) },
      { label: 'Full attention layers', value: fmtWNum(fullAttnLayerCount) },
      { label: 'Sparse attention layers', value: fmtWNum(sparseAttnLayerCount) },
      { label: 'Hidden size', value: fmtWNum(h) },
      { label: 'Attention heads', value: fmtWNum(n_q) },
      { label: 'KV heads', value: fmtWNum(h_kv) },
      { label: 'Head dim', value: fmtWNum(d_h) },
      { label: 'Attention per layer', value: fmtWNum(attnPerLayer) },
    ];
    if (sparseLayerCount > 0 && idxPerSparseLayer > 0) {
      breakdown.push({ label: 'Index heads', value: fmtWNum(idxHeads) });
      breakdown.push({ label: 'Index head dim', value: fmtWNum(idxHd) });
      breakdown.push({ label: 'Index projection per sparse layer', value: fmtWNum(idxPerSparseLayer) });
    }
    if (denseFfnCount > 0) {
      breakdown.push({ label: 'Dense FFN layers', value: fmtWNum(denseFfnCount) });
      breakdown.push({ label: 'Dense FFN per layer', value: fmtWNum(denseFfnPerLayer) });
    }
    if (moeFfnCount > 0) {
      breakdown.push({ label: 'MoE FFN layers', value: fmtWNum(moeFfnCount) });
      breakdown.push({ label: 'Routed experts', value: fmtWNum(nRouted) });
      breakdown.push({ label: 'Shared experts', value: fmtWNum(nShared) });
      breakdown.push({ label: 'Expert intermediate size', value: fmtWNum(Im) });
      breakdown.push({ label: 'Shared expert per layer', value: fmtWNum(sharedPerLayer) });
      breakdown.push({ label: 'Routed experts per layer', value: fmtWNum(expertPerLayer) });
    }
    breakdown.push({ label: 'Vocab size', value: fmtWNum(V) });
    breakdown.push({ label: 'Tie embeddings', value: tieEmbed ? 'Yes' : 'No' });
    breakdown.push({ label: 'Embedding params', value: fmtWNum(embedParams) });

  } else {
    formulaTitle = model.label + ' (unknown)';
  }

  if (visionParams > 0) {
    var vf = model.vision_fields || {};
    formulas.push({
      name: 'Vision',
      tip: (vf.estimated ? 'Estimated ' : '') + (vf.label || 'vision tower') + ' parameters. The selected weight precision is applied uniformly for planning.',
      expr: 'P_vision',
      values: { P_vision: visionParams },
      resultValue: visionParams,
      bar: [{ type: 'vision', bytes: visionParams * wtPrecB }],
      ibarVal: fmtWNum(visionParams)
    });
    breakdown.push({
      label: (vf.estimated ? 'Vision params (estimate)' : 'Vision params'),
      value: fmtWNum(visionParams),
      tip: vf.source || 'Auxiliary multimodal vision encoder parameters.'
    });
    patterns.push({ segs: [{ type: 'vision', ratio: 1 }], count: 1, label: vf.label || 'vision tower', bytes: visionParams * wtPrecB });
    if (legendTypes.indexOf('vision') === -1) legendTypes.push('vision');
  }

  if (engramParams > 0) {
    formulas.push({ name: 'Engram', tip: 'CPU-offloaded FP8 hash tables plus one scale byte per 32 elements, independent of selected transformer weight precision.', expr: 'P_engram', values: { P_engram: engramParams }, resultValue: engramParams, bar: [{ type: 'embed', bytes: engramBytes }], ibarVal: fmtWNum(engramParams) });
    patterns.push({ segs: [{ type: 'embed', ratio: 1 }], count: 1, label: 'Engram FP8 + scales (CPU RAM)', bytes: engramBytes });
  }
  var totalParams = attnParams + ffnDenseParams + ffnSharedParams + ffnExpertParams + embedParams + visionParams + engramParams;
  var totalBytes = (totalParams - engramParams) * wtPrecB + engramBytes;

  return {
    totalParams: totalParams,
    totalBytes: totalBytes,
    attnParams: attnParams,
    ffnDenseParams: ffnDenseParams,
    ffnSharedParams: ffnSharedParams,
    ffnExpertParams: ffnExpertParams,
    embedParams: embedParams,
    visionParams: visionParams,
    engramParams: engramParams,
    engramBytes: engramBytes,
    breakdown: breakdown,
    formulas: formulas,
    formulaTitle: formulaTitle,
    patterns: patterns,
    legendTypes: legendTypes,
  };
}
