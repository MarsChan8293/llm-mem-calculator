var GPU_OPTIONS = [
  { id: 'b300_288', label: 'B300 288GB', vram: 288 * 1e9 },
  { id: 'h20_141',        label: 'H20 141GB',         vram: 141 * 1e9 },
  { id: 'h200_141',       label: 'H200 141GB',        vram: 141 * 1e9 },
  { id: 'l20_48',         label: 'L20 48GB',          vram: 48 * 1e9 },
  { id: 'ascend_910b_64', label: 'Ascend 910B 64GB', vram: 64 * 1e9 },
  { id: 'ascend_950pr_112', label: 'Ascend 950PR 112GB', vram: 112 * 1e9 },
  { id: 'gb10_128',        label: 'GB10 128GB',        vram: 128 * 1e9 },
  { id: 'rtx_pro_5000_72', label: 'RTX PRO 5000 72GB', vram: 72 * 1e9 },
];

// vLLM deployment guardrails: reserve 10% of physical VRAM and a conservative
// per-GPU CUDA Graph capture budget before deciding whether a deployment fits.
var VLLM_GPU_MEMORY_UTILIZATION = 0.90;
var VLLM_CUDA_GRAPH_OVERHEAD_GB = 5;
var VLLM_CUDA_GRAPH_OVERHEAD_BYTES = VLLM_CUDA_GRAPH_OVERHEAD_GB * 1e9;

var DEPLOY_BAR_HEX_MAP = {
  'attn':        '#4263eb',
  'ffn-dense':   '#f59e0b',
  'ffn-shared':  '#e67700',
  'ffn-expert':  '#e03131',
  'embed':       '#9c36b5',
  'vision':      '#495057',
  'kv':          '#0c8599',
  'idx':         '#ae3ec9',
};

var DEPLOY_LEGEND_MAP = {
  'attn':        'Attention',
  'ffn-dense':   'Dense FFN',
  'ffn-shared':  'Shared Expert',
  'ffn-expert':  'Routed Experts',
  'embed':       'Embedding',
  'vision':      'Vision Tower',
  'kv':          'KV Cache',
  'idx':         'Indexer KV',
};

function calcDeploy(model, opts) {
  if (opts.mode === 'disaggregated') {
    var preOpts = Object.assign({}, opts, {
      tp: opts.prefill.tp,
      pp: opts.prefill.pp,
      ep: opts.prefill.ep,
      cp: opts.prefill.cp,
      dp: opts.prefill.dp,
    });
    var decOpts = Object.assign({}, opts, {
      tp: opts.decode.tp,
      pp: opts.decode.pp,
      ep: opts.decode.ep,
      cp: opts.decode.cp,
      dp: opts.decode.dp,
    });
    var preResult = calcDeployUnified(model, preOpts);
    var decResult = calcDeployUnified(model, decOpts);
    return {
      mode: 'disaggregated',
      prefill: preResult,
      decode: decResult,
      maxConcurrency: preResult.maxConcurrency === null || decResult.maxConcurrency === null
        ? null
        : Math.min(preResult.maxConcurrency, decResult.maxConcurrency),
      totalGPUs: {
        prefill: preResult.totalGPUs,
        decode: decResult.totalGPUs,
      },
    };
  }
  return calcDeployUnified(model, opts);
}

function sumLayerCacheBytes(layerBytes, startLayer, endLayer) {
  if (!Array.isArray(layerBytes)) return 0;
  var total = 0;
  for (var layer = startLayer; layer <= endLayer; layer++) total += layerBytes[layer] || 0;
  return total;
}

// Cache heads, not query heads, bound ordinary tensor-parallel sharding.
// For unsupported uneven head partitions use replication conservatively.
function cacheHeadSplit(heads, tp) {
  if (!heads) return 1;
  if (heads % tp === 0) return tp;
  if (tp % heads === 0) return heads;
  return 1;
}

function cacheTopology(model, opts) {
  var tp = opts.tp || 1;
  var f = model.fields;
  var linearTp = 1;
  if (model.formula === 'glm5_next_hybrid' && f.linear_num_heads % tp === 0) linearTp = tp;
  if (model.formula === 'qwen_qsa_gdn_hybrid' &&
      f.linear_num_key_heads % tp === 0 && f.linear_num_value_heads % tp === 0) linearTp = tp;
  var sharedIndexer = ['deepseek_v41', 'glm5_next_hybrid', 'qwen_qsa_gdn_hybrid'].includes(model.formula);
  return {
    kvTp: modelUsesMlaKv(model) ? 1 : cacheHeadSplit(f.num_key_value_heads, tp),
    linearTp: linearTp,
    idxTp: sharedIndexer ? 1 : (opts.idxTp || tp),
  };
}

function calcDeployUnified(model, opts) {
  var tp = opts.tp || 1;
  var pp = opts.pp || 1;
  var ep = opts.ep || 1;
  var dp = opts.dp || 1;
  var cp = opts.cp || 1;
  var topology = cacheTopology(model, opts);
  var idxTp = topology.idxTp;
  // MLA stores a shared latent KV representation. It is replicated across
  // tensor-parallel ranks and only partitioned along the context dimension.
  var kvTp = topology.kvTp;

  var f = model.fields;
  var wf = model.weight_fields || {};
  var L = f.num_hidden_layers;

  var weightResult = calcWeight(model, opts.wtPrecB);

  var denseLayerCount = 0, moeLayerCount = 0;
  for (var i = 0; i < L; i++) {
    if (isMoeLayer(wf, i)) moeLayerCount++;
    else denseLayerCount++;
  }

  var attnPerLayer = L > 0 ? weightResult.attnParams / L : 0;
  var denseFfnPerLayer = denseLayerCount > 0 ? weightResult.ffnDenseParams / denseLayerCount : 0;
  var sharedExpertPerLayer = moeLayerCount > 0 ? weightResult.ffnSharedParams / moeLayerCount : 0;
  var expertPerLayer = moeLayerCount > 0 ? weightResult.ffnExpertParams / moeLayerCount : 0;
  var embedTotal = weightResult.embedParams;
  var visionTotal = weightResult.visionParams || 0;

  var nRouted = wf.n_routed_experts || 0;
  var perExpertParams = nRouted > 0 && moeLayerCount > 0 ? weightResult.ffnExpertParams / (nRouted * moeLayerCount) : 0;

  var kvResult = calcKvCache(model, opts.tokens, opts.kvPrecB, opts.idxB, {
    seqs: 1,
    includeDraft: opts.includeDraft,
    includeLinear: opts.includeLinear,
  });
  var mainKvBytes = kvResult.mainKvBytes == null ? kvResult.kvBytes : kvResult.mainKvBytes;
  var kvPerLayerSingle = L > 0 ? mainKvBytes / L : 0;
  var idxPerLayerSingle = (kvResult.idxLayers || L) > 0 ? kvResult.idxBytes / (kvResult.idxLayers || L) : 0;
  var idxL = kvResult.idxLayers || L;

  var stages = [];
  var maxStageWeightPerGPU = 0;
  var maxStageTotalPerGPU = 0;

  for (var s = 0; s < pp; s++) {
    var startLayer = Math.floor(s * L / pp);
    var endLayer = Math.floor((s + 1) * L / pp) - 1;
    if (endLayer < startLayer) continue;
    if (startLayer > L - 1) break;
    var stageLayerCount = endLayer - startLayer + 1;

    var sDenseCount = 0, sMoeCount = 0;
    for (var li = startLayer; li <= endLayer; li++) {
      if (isMoeLayer(wf, li)) sMoeCount++;
      else sDenseCount++;
    }

    var sAttnPerGPU = stageLayerCount * attnPerLayer * opts.wtPrecB / tp;
    var sDenseFfnPerGPU = sDenseCount * denseFfnPerLayer * opts.wtPrecB / tp;
    var sSharedExpertPerGPU = sMoeCount * sharedExpertPerLayer * opts.wtPrecB / tp;
    var sRoutedExpertPerGPU = nRouted > 0
      ? Math.ceil(nRouted / ep) * perExpertParams * opts.wtPrecB * sMoeCount
      : 0;
    var sEmbedPerGPU = (s === 0 ? embedTotal * opts.wtPrecB / tp : 0);
    // Place the auxiliary vision encoder/aligner with the first pipeline
    // stage and shard its weights by TP for deployment planning.
    var sVisionPerGPU = (s === 0 ? visionTotal * opts.wtPrecB / tp : 0);

    var sWeightPerGPU = sAttnPerGPU + sDenseFfnPerGPU + sSharedExpertPerGPU + sRoutedExpertPerGPU + sEmbedPerGPU + sVisionPerGPU;

    var stageKvBytes = kvResult.kvLayerBytes
      ? sumLayerCacheBytes(kvResult.kvLayerBytes, startLayer, endLayer)
      : stageLayerCount * kvPerLayerSingle;
    // Draft/DSpark/MTP layers are appended after the main transformer and are
    // resident on the final PP stage in this calculator.
    if (s === pp - 1 && kvResult.draftKvBytes) stageKvBytes += kvResult.draftKvBytes;
    var stageIdxBytes = kvResult.idxLayerBytes
      ? sumLayerCacheBytes(kvResult.idxLayerBytes, startLayer, endLayer)
      : (stageLayerCount / L) * kvResult.idxBytes;
    var stageLinearBytes = sumLayerCacheBytes(kvResult.linearLayerBytes, startLayer, endLayer);
    // Recurrent/conv states have no token axis to divide by CP. Replicate
    // across CP ranks; shard whole linear heads only for divisible TP.
    var sKvPerGPU = ((stageKvBytes - stageLinearBytes) / (kvTp * cp)
      + stageLinearBytes / topology.linearTp) * opts.batch;
    var sIdxPerGPU = stageIdxBytes * opts.batch / (idxTp * cp);

    var sTotalPerGPU = sWeightPerGPU + sKvPerGPU + sIdxPerGPU;

    var ibarSegs = [];
    if (sAttnPerGPU > 0) ibarSegs.push({ type: 'attn', bytes: sAttnPerGPU });
    if (sDenseFfnPerGPU > 0) ibarSegs.push({ type: 'ffn-dense', bytes: sDenseFfnPerGPU });
    if (sSharedExpertPerGPU > 0) ibarSegs.push({ type: 'ffn-shared', bytes: sSharedExpertPerGPU });
    if (sRoutedExpertPerGPU > 0) ibarSegs.push({ type: 'ffn-expert', bytes: sRoutedExpertPerGPU });
    if (sEmbedPerGPU > 0) ibarSegs.push({ type: 'embed', bytes: sEmbedPerGPU });
    if (sVisionPerGPU > 0) ibarSegs.push({ type: 'vision', bytes: sVisionPerGPU });
    if (sKvPerGPU > 0) ibarSegs.push({ type: 'kv', bytes: sKvPerGPU });
    if (sIdxPerGPU > 0) ibarSegs.push({ type: 'idx', bytes: sIdxPerGPU });

    stages.push({
      stageIndex: s,
      layerRange: 'L' + startLayer + '-L' + endLayer,
      layerCount: stageLayerCount,
      denseLayers: sDenseCount,
      moeLayers: sMoeCount,
      attnPerGPU: sAttnPerGPU,
      denseFfnPerGPU: sDenseFfnPerGPU,
      sharedExpertPerGPU: sSharedExpertPerGPU,
      routedExpertPerGPU: sRoutedExpertPerGPU,
      embedPerGPU: sEmbedPerGPU,
      visionPerGPU: sVisionPerGPU,
      kvPerGPU: sKvPerGPU,
      idxPerGPU: sIdxPerGPU,
      weightPerGPU: sWeightPerGPU,
      totalPerGPU: sTotalPerGPU,
      ibar: ibarSegs,
    });

    if (sWeightPerGPU > maxStageWeightPerGPU) maxStageWeightPerGPU = sWeightPerGPU;
    if (sTotalPerGPU > maxStageTotalPerGPU) maxStageTotalPerGPU = sTotalPerGPU;
  }

  var bottleneckStage = stages.reduce(function (a, b) { return a.totalPerGPU > b.totalPerGPU ? a : b; });

  var weightBreakdown = {
    attnPerGPU: bottleneckStage.attnPerGPU,
    denseFfnPerGPU: bottleneckStage.denseFfnPerGPU,
    sharedExpertPerGPU: bottleneckStage.sharedExpertPerGPU,
    routedExpertPerGPU: bottleneckStage.routedExpertPerGPU,
    embedPerGPU: bottleneckStage.embedPerGPU,
    visionPerGPU: bottleneckStage.visionPerGPU || 0,
  };
  var kvBreakdown = {
    kvPerGPU: bottleneckStage.kvPerGPU,
    idxPerGPU: bottleneckStage.idxPerGPU,
  };

  var gpuOption = GPU_OPTIONS.find(function (g) { return g.id === opts.gpuId; }) || GPU_OPTIONS[0];
  var gpuVram = gpuOption.vram;
  var gpuUsableVram = gpuVram * VLLM_GPU_MEMORY_UTILIZATION;
  var gpuUsedWithOverhead = maxStageTotalPerGPU + VLLM_CUDA_GRAPH_OVERHEAD_BYTES;
  var gpuUsage = gpuUsedWithOverhead / gpuUsableVram;
  var gpuFits = gpuUsage <= 1.0;

  // Estimate the largest number of selected-context sequences that can fit
  // after weights, the 90% vLLM budget, and the fixed per-GPU reserve. The
  // minimum across pipeline stages is the safe limit for PP deployments.
  var batch = Math.max(1, opts.batch || 1);
  var maxConcurrency = null;
  var kvSpacePerGPU = null;
  var concurrencyBottleneck = null;
  stages.forEach(function (stage) {
    var stageKvPerSequence = (stage.kvPerGPU + stage.idxPerGPU) / batch;
    if (stageKvPerSequence <= 0) return;
    var stageKvBudget = Math.max(0, gpuUsableVram - VLLM_CUDA_GRAPH_OVERHEAD_BYTES - stage.weightPerGPU);
    kvSpacePerGPU = kvSpacePerGPU === null
      ? stageKvBudget
      : Math.min(kvSpacePerGPU, stageKvBudget);
    var stageMaxConcurrency = Math.floor(stageKvBudget / stageKvPerSequence);
    if (maxConcurrency === null || stageMaxConcurrency < maxConcurrency) {
      maxConcurrency = stageMaxConcurrency;
      concurrencyBottleneck = {
        stageIndex: stage.stageIndex,
        layerRange: stage.layerRange,
        kvBudget: stageKvBudget,
        kvPerSequence: stageKvPerSequence,
        exactConcurrency: stageKvBudget / stageKvPerSequence,
      };
    }
  });
  var bottleneckKvPerSequence = (bottleneckStage.kvPerGPU + bottleneckStage.idxPerGPU) / batch;

  var formulaTitle = model.label + ' per-GPU (TP=' + tp + ', PP=' + pp + ', EP=' + ep + (cp > 1 ? ', CP=' + cp : '') + ')';
  var formulas = buildDeployFormulas(model, opts, weightResult, kvResult, stages);

  var ibarSegments = bottleneckStage.ibar;
  var legendTypes = ibarSegments.map(function (seg) { return seg.type; });

  return {
    mode: 'unified',
    weightPerGPU: maxStageWeightPerGPU,
    kvPerGPU: bottleneckStage.kvPerGPU + bottleneckStage.idxPerGPU,
    kvPerGPUPerSequence: bottleneckKvPerSequence,
    kvTpSplit: kvTp,
    kvCpSplit: cp,
    idxTpSplit: idxTp,
    cacheNote: 'Attention KV uses effective TP split=' + kvTp + ' and CP=' + cp
      + '. Fixed linear state, where included, uses TP split=' + topology.linearTp
      + ' and is not divided by CP. Single-key indexers use replicated TP storage. '
      + (['glm5_next_hybrid', 'qwen_qsa_gdn_hybrid'].includes(model.formula)
        ? 'Provisional estimate: indexer tail/MTP layout and backend topology support require verification. '
        : '')
      + (model.formula === 'deepseek_v41' && pp > 1
        ? 'Source-owned cache sharing across PP stages requires backend support. ' : ''),
    kvSpacePerGPU: kvSpacePerGPU === null ? 0 : kvSpacePerGPU,
    maxConcurrency: maxConcurrency,
    concurrencyBottleneck: concurrencyBottleneck,
    totalPerGPU: maxStageTotalPerGPU,
    totalGPUs: tp * pp * dp,
    weightBreakdown: weightBreakdown,
    kvBreakdown: kvBreakdown,
    stages: stages,
    bottleneckStageIndex: bottleneckStage.stageIndex,
    gpuFit: {
      gpuId: gpuOption.id,
      label: gpuOption.label,
      vram: gpuVram,
      usableVram: gpuUsableVram,
      fixedOverhead: VLLM_CUDA_GRAPH_OVERHEAD_BYTES,
      usedVram: gpuUsedWithOverhead,
      utilizationLimit: VLLM_GPU_MEMORY_UTILIZATION,
      usage: gpuUsage,
      fits: gpuFits,
    },
    formulas: formulas,
    formulaTitle: formulaTitle,
    ibarSegments: ibarSegments,
    legendTypes: legendTypes,
  };
}

function buildDeployFormulas(model, opts, weightResult, kvResult, stages) {
  var tp = opts.tp || 1;
  var ep = opts.ep || 1;
  var topology = cacheTopology(model, opts);
  var idxTp = topology.idxTp;
  var cp = opts.cp || 1;
  var kvTp = topology.kvTp;
  var wf = model.weight_fields || {};
  var f = model.fields;
  var L = f.num_hidden_layers;
  var batch = opts.batch;
  var wtPrecB = opts.wtPrecB;

  var nRouted = wf.n_routed_experts || 0;
  var nShared = wf.n_shared_experts || 0;
  var h = wf.hidden_size || 0;
  var V = wf.vocab_size || 0;
  var tieEmbed = wf.tie_word_embeddings;

  var attnPerLayer = L > 0 ? weightResult.attnParams / L : 0;
  var denseLayerCount = 0, moeLayerCount = 0;
  for (var i = 0; i < L; i++) {
    if (isMoeLayer(wf, i)) moeLayerCount++;
    else denseLayerCount++;
  }
  var denseFfnPerLayer = denseLayerCount > 0 ? weightResult.ffnDenseParams / denseLayerCount : 0;
  var sharedPerLayer = moeLayerCount > 0 ? weightResult.ffnSharedParams / moeLayerCount : 0;
  var expertPerLayer = moeLayerCount > 0 ? weightResult.ffnExpertParams / moeLayerCount : 0;
  var perExpertParams = nRouted > 0 && moeLayerCount > 0 ? weightResult.ffnExpertParams / (nRouted * moeLayerCount) : 0;

  var mainKvBytes = kvResult.mainKvBytes == null ? kvResult.kvBytes : kvResult.mainKvBytes;
  var kvPerLayerSingle = L > 0 ? mainKvBytes / L : 0;
  var idxPerLayerSingle = (kvResult.idxLayers || L) > 0 ? kvResult.idxBytes / (kvResult.idxLayers || L) : 0;
  var idxL = kvResult.idxLayers || L;

  var linearBytes = sumLayerCacheBytes(kvResult.linearLayerBytes, 0, L - 1);
  kvPerLayerSingle = (mainKvBytes - linearBytes) / L;
  var formulas = [];
  if (linearBytes > 0) {
    var linearPerGPU = linearBytes * batch / topology.linearTp;
    formulas.push({
      name: 'Linear state/linear_tp',
      tip: 'Fixed convolution and FP32 recurrent state: whole-head TP sharding when divisible; replicated across CP. Backend support must be verified.',
      expr: 'State×B/linear_tp',
      values: { State: fmtWBytes(linearBytes), B: batch, linear_tp: topology.linearTp },
      resultValue: linearPerGPU,
      bar: [{ type: 'kv', bytes: linearPerGPU }],
      ibarVal: fmtWBytes(linearPerGPU),
    });
  }

  formulas.push({
    name: 'Attn/tp',
    tip: 'Attention weights per layer, split by TP.',
    expr: 'Attn\u00d7L/tp',
    values: { Attn: fmtWNum(Math.round(attnPerLayer)), L: L, tp: tp },
    resultValue: attnPerLayer * L * wtPrecB / tp,
    bar: [{ type: 'attn', bytes: attnPerLayer * L * wtPrecB / tp }],
    ibarVal: fmtWBytes(attnPerLayer * L * wtPrecB / tp),
  });

  if (denseLayerCount > 0) {
    formulas.push({
      name: 'FFN_d/tp',
      tip: 'Dense FFN weights, split by TP.',
      expr: 'FFN_d\u00d7L_d/tp',
      values: { FFN_d: fmtWNum(Math.round(denseFfnPerLayer)), L_d: denseLayerCount, tp: tp },
      resultValue: denseFfnPerLayer * denseLayerCount * wtPrecB / tp,
      bar: [{ type: 'ffn-dense', bytes: denseFfnPerLayer * denseLayerCount * wtPrecB / tp }],
      ibarVal: fmtWBytes(denseFfnPerLayer * denseLayerCount * wtPrecB / tp),
    });
  }

  if (nShared > 0) {
    formulas.push({
      name: 'FFN_s/tp',
      tip: 'Shared expert weights per MoE layer, split by TP.',
      expr: 'FFN_s\u00d7L_m/tp',
      values: { FFN_s: fmtWNum(Math.round(sharedPerLayer)), L_m: moeLayerCount, tp: tp },
      resultValue: sharedPerLayer * moeLayerCount * wtPrecB / tp,
      bar: [{ type: 'ffn-shared', bytes: sharedPerLayer * moeLayerCount * wtPrecB / tp }],
      ibarVal: fmtWBytes(sharedPerLayer * moeLayerCount * wtPrecB / tp),
    });
  }

  if (nRouted > 0) {
    var expertCountPerGPU = Math.ceil(nRouted / ep);
    formulas.push({
      name: 'FFN_e/ep',
      tip: 'Routed expert weights, split by EP.',
      expr: '\u2308N_e/ep\u2309\u00d7FFN_e\u00d7L_m',
      values: { N_e: nRouted, ep: ep, FFN_e: fmtWNum(Math.round(perExpertParams)), L_m: moeLayerCount },
      resultValue: expertCountPerGPU * perExpertParams * moeLayerCount * wtPrecB,
      bar: [{ type: 'ffn-expert', bytes: expertCountPerGPU * perExpertParams * moeLayerCount * wtPrecB }],
      ibarVal: fmtWBytes(expertCountPerGPU * perExpertParams * moeLayerCount * wtPrecB),
    });
  }

  formulas.push({
    name: 'Embed/tp',
    tip: tieEmbed ? 'Embedding only (tied), split by TP.' : 'Embedding + lm_head, split by TP.',
    expr: tieEmbed ? 'V\u00d7h/tp' : '2\u00d7V\u00d7h/tp',
    values: { V: V, h: h, tp: tp },
    resultValue: weightResult.embedParams * wtPrecB / tp,
    bar: [{ type: 'embed', bytes: weightResult.embedParams * wtPrecB / tp }],
    ibarVal: fmtWBytes(weightResult.embedParams * wtPrecB / tp),
  });

  if ((weightResult.visionParams || 0) > 0) {
    var vf = model.vision_fields || {};
    formulas.push({
      name: 'Vision/tp',
      tip: (vf.estimated ? 'Estimated ' : '') + (vf.label || 'vision tower') + ' weights placed on the first PP stage and split by TP for planning.',
      expr: 'P_vision×p/tp',
      values: { P_vision: fmtWNum(weightResult.visionParams), p: wtPrecB, tp: tp },
      resultValue: weightResult.visionParams * wtPrecB / tp,
      bar: [{ type: 'vision', bytes: weightResult.visionParams * wtPrecB / tp }],
      ibarVal: fmtWBytes(weightResult.visionParams * wtPrecB / tp),
    });
  }

  formulas.push({
    name: 'Attention KV/(kv_tp×cp)',
    tip: modelUsesMlaKv(model)
      ? 'MLA KV is replicated across TP ranks and split along the context dimension by CP.'
      : 'Attention KV uses effective KV-head TP splitting, capped by KV heads; fixed linear state is separate.',
    expr: modelUsesMlaKv(model) ? 'KV\u00d7L\u00d7B/cp' : 'KV\u00d7L\u00d7B/(tp\u00d7cp)',
    values: modelUsesMlaKv(model)
      ? { KV: fmtWBytes(kvPerLayerSingle), L: L, B: batch, cp: cp }
      : { KV: fmtWBytes(kvPerLayerSingle), L: L, B: batch, tp: kvTp, cp: cp },
    resultValue: kvPerLayerSingle * L * batch / (kvTp * cp),
    bar: [{ type: 'kv', bytes: kvPerLayerSingle * L * batch / (kvTp * cp) }],
    ibarVal: fmtWBytes(kvPerLayerSingle * L * batch / (kvTp * cp)),
  });

  if (kvResult.draftKvBytes > 0) {
    formulas.push({
      name: 'KV_draft',
      tip: 'Optional draft/MTP cache placed on the final pipeline stage.',
      expr: 'KV_draft×B/(tp×cp)',
      values: { KV_draft: fmtWBytes(kvResult.draftKvBytes), B: batch, tp: kvTp, cp: cp },
      resultValue: kvResult.draftKvBytes * batch / (kvTp * cp),
      bar: [{ type: 'kv', bytes: kvResult.draftKvBytes * batch / (kvTp * cp) }],
      ibarVal: fmtWBytes(kvResult.draftKvBytes * batch / (kvTp * cp)),
    });
  }

  if (idxPerLayerSingle > 0) {
    var idxExpr = 'Idx\u00d7' + (idxL !== L ? 'L_idx' : 'L') + '\u00d7B/(tp_idx\u00d7cp)';
    var idxValues = idxL !== L
      ? { Idx: fmtWBytes(idxPerLayerSingle), L_idx: idxL, B: batch, tp_idx: idxTp, cp: cp }
      : { Idx: fmtWBytes(idxPerLayerSingle), L: L, B: batch, tp_idx: idxTp, cp: cp };
    formulas.push({
      name: 'Idx/tp_idx',
      tip: idxL !== L ? 'Indexer KV cache per indexer layer, split by Indexer TP. With IndexShare, only ' + idxL + ' of ' + L + ' layers have indexer.' : 'Indexer KV cache per layer, split by Indexer TP, times batch.',
      expr: idxExpr,
      values: idxValues,
      resultValue: idxPerLayerSingle * idxL * batch / (idxTp * cp),
      bar: [{ type: 'idx', bytes: idxPerLayerSingle * idxL * batch / (idxTp * cp) }],
      ibarVal: fmtWBytes(idxPerLayerSingle * idxL * batch / (idxTp * cp)),
    });
  }

  return formulas;
}

function getDeployDefaults(model) {
  var wf = model.weight_fields || {};
  var nRouted = wf.n_routed_experts || 0;
  var isMoE = nRouted > 0;
  var h = wf.hidden_size || 0;

  var wr = calcWeight(model, 2);
  var totalB = wr.totalParams / 1e9;

  var tp, gpu;
  if (totalB >= 400 || (isMoE && totalB >= 100)) {
    tp = 8;
    gpu = 'b300_288';
  } else if (totalB >= 30 || h >= 4096) {
    tp = 4;
    gpu = 'h200_141';
  } else if (totalB >= 7 || h >= 2048) {
    tp = 2;
    gpu = 'l20_48';
  } else {
    tp = 1;
    gpu = 'l20_48';
  }

  var ep = isMoE ? tp : 1;
  var pp = 1;
  var dp = 1;

  return { tp: tp, pp: pp, ep: ep, dp: dp, idxTp: tp, gpu: gpu };
}

function modelHasIndexer(model) {
  var formula = model.formula;
  return formula === 'deepseek_v4_hybrid' || formula === 'deepseek_v41' || formula === 'dsa_mla' || formula === 'msa_gqa' || formula === 'glm5_next_hybrid' || formula === 'qwen_qsa_gdn_hybrid';
}

function modelUsesMlaKv(model) {
  // DeepSeek V4's hybrid sliding/compressed cache is still a latent KV
  // payload: it is replicated across TP ranks and only sharded by CP.
  return ['mla', 'dsa_mla', 'deepseek_v4_hybrid', 'deepseek_v41', 'glm5_next_hybrid', 'kda_gated_mla'].includes(model.formula);
}

function modelSupportsAbsorption(model) {
  var formula = model.formula;
  return formula === 'mla' || formula === 'dsa_mla' || formula === 'deepseek_v4_hybrid' || formula === 'deepseek_v41' || formula === 'glm5_next_hybrid' || formula === 'kda_gated_mla';
}
