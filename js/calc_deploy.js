var GPU_OPTIONS = [
  { id: 'h100_80',  label: 'H100 80GiB',  vram: 80 * Math.pow(1024, 3) },
  { id: 'h200_141', label: 'H200 141GiB', vram: 141 * Math.pow(1024, 3) },
  { id: 'b200_180', label: 'B200 180GiB', vram: 180 * Math.pow(1024, 3) },
  { id: 'a100_80',  label: 'A100 80GiB',  vram: 80 * Math.pow(1024, 3) },
  { id: 'a100_40',  label: 'A100 40GiB',  vram: 40 * Math.pow(1024, 3) },
  { id: 'h100_40',  label: 'H100 40GiB',  vram: 40 * Math.pow(1024, 3) },
  { id: 'l40s_48',  label: 'L40S 48GiB',  vram: 48 * Math.pow(1024, 3) },
  { id: 'l4_24',    label: 'L4 24GiB',    vram: 24 * Math.pow(1024, 3) },
];

var DEPLOY_BAR_HEX_MAP = {
  'attn':        '#4263eb',
  'ffn-dense':   '#f59e0b',
  'ffn-shared':  '#e67700',
  'ffn-expert':  '#e03131',
  'embed':       '#9c36b5',
  'kv':          '#0c8599',
  'idx':         '#ae3ec9',
};

var DEPLOY_LEGEND_MAP = {
  'attn':        'Attention',
  'ffn-dense':   'Dense FFN',
  'ffn-shared':  'Shared Expert',
  'ffn-expert':  'Routed Experts',
  'embed':       'Embedding',
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
      totalGPUs: {
        prefill: preResult.totalGPUs,
        decode: decResult.totalGPUs,
      },
    };
  }
  return calcDeployUnified(model, opts);
}

function calcDeployUnified(model, opts) {
  var tp = opts.tp || 1;
  var pp = opts.pp || 1;
  var ep = opts.ep || 1;
  var dp = opts.dp || 1;
  var cp = opts.cp || 1;
  var idxTp = opts.idxTp || tp;

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

  var nRouted = wf.n_routed_experts || 0;
  var perExpertParams = nRouted > 0 && moeLayerCount > 0 ? weightResult.ffnExpertParams / (nRouted * moeLayerCount) : 0;

  var kvResult = calcKvCache(model, opts.tokens, opts.kvPrecB, opts.idxB, {
    seqs: 1,
    includeDraft: opts.includeDraft,
    includeLinear: opts.includeLinear,
  });
  var kvPerLayerSingle = L > 0 ? kvResult.kvBytes / L : 0;
  var idxPerLayerSingle = (kvResult.idxLayers || L) > 0 ? kvResult.idxBytes / (kvResult.idxLayers || L) : 0;
  var idxL = kvResult.idxLayers || L;

  var layersPerStage = Math.ceil(L / pp);
  var stages = [];
  var maxStageWeightPerGPU = 0;
  var maxStageTotalPerGPU = 0;

  for (var s = 0; s < pp; s++) {
    var startLayer = s * layersPerStage;
    var endLayer = Math.min((s + 1) * layersPerStage - 1, L - 1);
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

    var sWeightPerGPU = sAttnPerGPU + sDenseFfnPerGPU + sSharedExpertPerGPU + sRoutedExpertPerGPU + sEmbedPerGPU;

    var sKvPerGPU = stageLayerCount * kvPerLayerSingle * opts.batch / (tp * cp);
    var sIdxPerGPU = (stageLayerCount / L) * kvResult.idxBytes * opts.batch / (idxTp * cp);

    var sTotalPerGPU = sWeightPerGPU + sKvPerGPU + sIdxPerGPU;

    var ibarSegs = [];
    if (sAttnPerGPU > 0) ibarSegs.push({ type: 'attn', bytes: sAttnPerGPU });
    if (sDenseFfnPerGPU > 0) ibarSegs.push({ type: 'ffn-dense', bytes: sDenseFfnPerGPU });
    if (sSharedExpertPerGPU > 0) ibarSegs.push({ type: 'ffn-shared', bytes: sSharedExpertPerGPU });
    if (sRoutedExpertPerGPU > 0) ibarSegs.push({ type: 'ffn-expert', bytes: sRoutedExpertPerGPU });
    if (sEmbedPerGPU > 0) ibarSegs.push({ type: 'embed', bytes: sEmbedPerGPU });
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
  };
  var kvBreakdown = {
    kvPerGPU: bottleneckStage.kvPerGPU,
    idxPerGPU: bottleneckStage.idxPerGPU,
  };

  var gpuOption = GPU_OPTIONS.find(function (g) { return g.id === opts.gpuId; }) || GPU_OPTIONS[0];
  var gpuVram = gpuOption.vram;
  var gpuUsage = maxStageTotalPerGPU / gpuVram;
  var gpuFits = gpuUsage <= 1.0;

  var formulaTitle = model.label + ' per-GPU (TP=' + tp + ', PP=' + pp + ', EP=' + ep + (cp > 1 ? ', CP=' + cp : '') + ')';
  var formulas = buildDeployFormulas(model, opts, weightResult, kvResult, stages);

  var ibarSegments = bottleneckStage.ibar;
  var legendTypes = ibarSegments.map(function (seg) { return seg.type; });

  return {
    mode: 'unified',
    weightPerGPU: maxStageWeightPerGPU,
    kvPerGPU: bottleneckStage.kvPerGPU + bottleneckStage.idxPerGPU,
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
  var idxTp = opts.idxTp || tp;
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

  var kvPerLayerSingle = L > 0 ? kvResult.kvBytes / L : 0;
  var idxPerLayerSingle = (kvResult.idxLayers || L) > 0 ? kvResult.idxBytes / (kvResult.idxLayers || L) : 0;
  var idxL = kvResult.idxLayers || L;

  var formulas = [];

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

  formulas.push({
    name: 'KV/tp',
    tip: 'KV cache per layer, split by TP, times batch.',
    expr: 'KV\u00d7L\u00d7B/tp',
    values: { KV: fmtWBytes(kvPerLayerSingle), L: L, B: batch, tp: tp },
    resultValue: kvPerLayerSingle * L * batch / tp,
    bar: [{ type: 'kv', bytes: kvPerLayerSingle * L * batch / tp }],
    ibarVal: fmtWBytes(kvPerLayerSingle * L * batch / tp),
  });

  if (idxPerLayerSingle > 0) {
    var idxExpr = idxL !== L ? 'Idx\u00d7L_idx\u00d7B/tp_idx' : 'Idx\u00d7L\u00d7B/tp_idx';
    var idxValues = idxL !== L
      ? { Idx: fmtWBytes(idxPerLayerSingle), L_idx: idxL, B: batch, tp_idx: idxTp }
      : { Idx: fmtWBytes(idxPerLayerSingle), L: L, B: batch, tp_idx: idxTp };
    formulas.push({
      name: 'Idx/tp_idx',
      tip: idxL !== L ? 'Indexer KV cache per indexer layer, split by Indexer TP. With IndexShare, only ' + idxL + ' of ' + L + ' layers have indexer.' : 'Indexer KV cache per layer, split by Indexer TP, times batch.',
      expr: idxExpr,
      values: idxValues,
      resultValue: idxPerLayerSingle * idxL * batch / idxTp,
      bar: [{ type: 'idx', bytes: idxPerLayerSingle * idxL * batch / idxTp }],
      ibarVal: fmtWBytes(idxPerLayerSingle * idxL * batch / idxTp),
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
    gpu = 'h100_80';
  } else if (totalB >= 30 || h >= 4096) {
    tp = 4;
    gpu = 'h100_80';
  } else if (totalB >= 7 || h >= 2048) {
    tp = 2;
    gpu = 'a100_40';
  } else {
    tp = 1;
    gpu = 'a100_40';
  }

  var ep = isMoE ? tp : 1;
  var pp = 1;
  var dp = 1;

  return { tp: tp, pp: pp, ep: ep, dp: dp, idxTp: tp, gpu: gpu };
}

function modelHasIndexer(model) {
  var formula = model.formula;
  return formula === 'deepseek_v4_hybrid' || formula === 'dsa_mla' || formula === 'msa_gqa';
}

function modelSupportsAbsorption(model) {
  var formula = model.formula;
  return formula === 'mla' || formula === 'dsa_mla' || formula === 'deepseek_v4_hybrid';
}
