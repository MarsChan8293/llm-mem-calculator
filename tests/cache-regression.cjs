const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({});
for (const file of ['data', 'calc', 'calc_weight', 'calc_deploy']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../js', file + '.js'), 'utf8'), context);
}
const models = vm.runInContext('MODEL_DATA.models', context);
const base = { tokens: 1024, kvPrecB: 2, idxB: 1, wtPrecB: 1, batch: 1,
  tp: 1, cp: 1, pp: 1, ep: 1, includeLinear: true };
function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < Math.max(1e-6, Math.abs(expected) * 1e-12), `${actual} != ${expected}`);
}
for (const [id, state, heads] of [
  ['glm-5.3-flash', 34 * (4 * 3 * 64 * 128 * 2 + 64 * 128 * 128 * 4), 1],
  ['qwen3.8-flash-next', 36 * (4 * (2 * 16 * 128 + 48 * 128) * 2 + 48 * 128 * 128 * 4), 2],
]) {
  const model = models.find(m => m.id === id);
  for (const tp of [1, 2, 8]) for (const cp of [1, 4]) for (const pp of [1, 8, 16]) {
    const opts = { ...base, tp, cp, pp, includeDraft: true };
    const withState = context.calcDeploy(model, opts);
    const withoutState = context.calcDeploy(model, { ...opts, includeLinear: false });
    const sum = r => r.stages.reduce((n, s) => n + s.kvPerGPU, 0);
    near(sum(withState) - sum(withoutState), state / tp);
    assert.equal(withState.kvTpSplit, Math.min(tp, heads));
    assert.equal(withState.idxTpSplit, 1);
    assert.equal(withState.stages.length, pp);
    const cache = context.calcKvCache(model, base.tokens, 2, 1, { includeDraft: true });
    near(sum(withoutState), cache.kvBytes / (Math.min(tp, heads) * cp));
    const formulaCache = withState.formulas.filter(f => f.bar.every(b => ['kv', 'idx'].includes(b.type)))
      .reduce((n, f) => n + f.resultValue, 0);
    near(formulaCache, withState.stages.reduce((n, s) => n + s.kvPerGPU + s.idxPerGPU, 0));
  }
}
const ds = models.find(m => m.id === 'deepseek-v4.1-flash');
// Independent native-layout arithmetic: three half-rate owners plus one
// full-rate owner; FP4 values and one-byte per-group scales.
for (const tokens of [1, 3, 1024, 1048576]) {
  for (const kvB of [0.5, 1, 2]) for (const idxB of [0.5, 1, 2]) {
    const slots = 3 * Math.floor(tokens / 2) + tokens;
    const kvSlot = 512 * kvB + (kvB === 0.5 ? 32 : 0);
    const idxSlot = 128 * idxB + (idxB === 0.5 ? 4 : 0);
    const result = context.calcKvCache(ds, tokens, kvB, idxB, {});
    assert.equal(result.idxLayers, 4);
    near(result.idxBytes, slots * idxSlot);
    near(result.globalCacheBytes, slots * (kvSlot + idxSlot));
    near(result.kvBytes + result.idxBytes, result.globalCacheBytes + 40 * 128 * 512 * kvB);
    assert.equal(result.idxLayerBytes[24], undefined);
  }
}
const native = context.calcKvCache(ds, 1048576, 0.5, 0.5, {});
near(native.globalCacheBytes / 1048576, 890);
const nativeDraft = context.calcKvCache(ds, 1048576, 0.5, 0.5, { includeDraft: true });
near(native.globalCacheBytes, nativeDraft.globalCacheBytes);
for (const pp of [1, 8, 16]) {
  const a = context.calcDeploy(ds, { ...base, pp, includeDraft: true });
  const b = context.calcDeploy(ds, { ...base, pp, includeDraft: false });
  assert.equal(a.stages.length, pp);
  near(a.stages.at(-1).kvPerGPU - b.stages.at(-1).kvPerGPU, 3 * 128 * 512 * 2);
  assert.equal(a.stages.reduce((n, s) => n + s.layerCount, 0), 40);
}
// DeepSeek V4 configs append Draft/MTP compress ratios after backbone layers.
for (const [id, draftCount] of [
  ['deepseek-v4-pro', 1],
  ['deepseek-v4-pro-0813', 1],
  ['deepseek-v4-flash', 1],
  ['deepseek-v4-flash-0731', 3],
  ['deepseek-v4-flash-vision-exp', 3],
]) {
  const model = models.find(m => m.id === id);
  assert.ok(model, id + ' missing');
  assert.equal(model.fields.num_nextn_predict_layers, draftCount, id + ' draft count');
  assert.equal(model.fields.compress_ratios.length, model.fields.num_hidden_layers + draftCount, id + ' ratio layout');
  const noDraft = context.calcKvCache(model, 1024, 2, 1, { includeDraft: false });
  const withDraft = context.calcKvCache(model, 1024, 2, 1, { includeDraft: true });
  const expectedDraft = draftCount * 128 * 512 * 2;
  near(withDraft.draftKvBytes, expectedDraft);
  near(withDraft.kvBytes - noDraft.kvBytes, expectedDraft);
  const mainRatios = Array.from(model.fields.compress_ratios).slice(0, model.fields.num_hidden_layers);
  assert.equal(withDraft.idxLayers, mainRatios.filter(r => r === 4).length);
  assert.equal(withDraft.kvLayerBytes.length, model.fields.num_hidden_layers);
  assert.equal(withDraft.idxLayerBytes.length, model.fields.num_hidden_layers);
}
// In particular, the preview Flash backbone begins with two r=0 layers but has
// only one appended Draft layer; leading r=0 target layers are not Draft.
assert.equal(models.find(m => m.id === 'deepseek-v4-flash').fields.compress_ratios.slice(0, 43).filter(r => r === 0).length, 2);
near(context.calcKvCache(models.find(m => m.id === 'deepseek-v4-flash'), 1024, 2, 1, { includeDraft: true }).draftKvBytes, 128 * 512 * 2);

for (const [id, visionParams] of [
  ['deepseek-v4-flash-vision-exp', 466376704],
  ['deepseek-v4.1-flash', 485268480],
  ['kimi-k2.7-code', 400000000],
  ['glm-5.3-flash', 530131968],
  ['qwen3.8-27b', 460730096],
  ['qwen3.8-flash-next', 448931056],
]) {
  const model = models.find(m => m.id === id);
  assert.ok(model, id + ' missing');
  const weight = context.calcWeight(model, 1);
  assert.equal(weight.visionParams, visionParams, id + ' vision params');
  const opts = { ...base, tp: 2, pp: 2, ep: 1, includeDraft: true, gpuId: 'b300_288' };
  const withVision = context.calcDeploy(model, opts);
  const withoutVision = context.calcDeploy({ ...model, vision_fields: null }, opts);
  near(withVision.stages[0].weightPerGPU - withoutVision.stages[0].weightPerGPU, visionParams / 2);
  near(withVision.stages[1].weightPerGPU - withoutVision.stages[1].weightPerGPU, 0);
}


// DeepSeek V4.1 official config: Engram offload, DSpark draft, and vision.
assert.equal(ds.fields.engram_num_embeddings[0], 384006168);
assert.equal(ds.fields.engram_num_embeddings[1], 384016682);
assert.equal(ds.fields.engram_head_dim, 256);
assert.equal(ds.fields.dspark_block_size, 5);
assert.equal(ds.fields.dspark_n_routed_experts, 128);
assert.equal(ds.fields.dspark_num_experts_per_tok, 3);
assert.equal(ds.fields.dspark_target_layer_ids.join(','), '37,38,39');
const engramParams = (384006168 + 384016682) * 256;
const v41VisionParams = (3 * 14 * 14 * 1024 + 1024)
  + 32 * (2 * 1024 + (4 * 1024 * 1024 + 4 * 1024) + (3 * 1024 * 2816))
  + 1024
  + (1024 * 9 * 5120 + 5120) + (5120 * 5120 + 5120)
  + 3 * 5120;
assert.equal(v41VisionParams, 485268480);
const v41Weight = context.calcWeight(ds, 1);
assert.equal(v41Weight.visionParams, v41VisionParams);
assert.equal(v41Weight.totalParams - context.calcWeight({ ...ds, vision_fields: null }, 1).totalParams, v41VisionParams);
assert.ok(v41Weight.breakdown.some(item => item.label === 'Engram table params (FP8 resident)' && item.value === engramParams.toLocaleString('en-US')));
const withoutEngram = { ...ds, fields: { ...ds.fields, engram_num_embeddings: [], engram_layer_ids: [] } };
const v41NoEngram = context.calcWeight(withoutEngram, 1);
near(v41Weight.totalParams - v41NoEngram.totalParams, engramParams);
near(v41Weight.totalBytes - v41NoEngram.totalBytes, engramParams * 33 / 32);
for (const pp of [1, 2, 8]) {
  const a = context.calcDeploy(ds, { ...base, pp, tp: 2 });
  const b = context.calcDeploy(withoutEngram, { ...base, pp, tp: 2 });
  for (let i = 0; i < pp; i++) {
    const lo = Math.floor(i * 40 / pp), hi = Math.floor((i + 1) * 40 / pp) - 1;
    const expected = ds.fields.engram_layer_ids.reduce((n, layer, j) => n + (layer >= lo && layer <= hi ? Math.ceil(ds.fields.engram_num_embeddings[j] / 2) * 256 * 33 / 32 : 0), 0);
    near(a.stages[i].weightPerGPU - b.stages[i].weightPerGPU, expected);
    near(a.stages[i].engramPerGPU, expected);
  }
}

const agentWorld = models.find(m => m.id === 'qwen-agentworld-35b-a3b');
assert.ok(agentWorld, 'Qwen AgentWorld checkpoint missing');
assert.equal(context.calcWeight(agentWorld, 1).visionParams, 0, 'AgentWorld checkpoint is language-model-only');

const glm = models.find(m => m.id === 'glm-5.3-flash');
const zeroRank = { ...glm, fields: { ...glm.fields, q_lora_rank: 0 } };
near(context.calcWeight(glm, 1).attnParams - context.calcWeight(zeroRank, 1).attnParams, 415236096);
for (const model of models) {
  const result = context.calcDeploy(model, base);
  assert.ok(Number.isFinite(result.totalPerGPU) && result.totalPerGPU > 0, model.id);
}
console.log(`Cache topology regression tests passed; ${models.length} model smoke checks passed.`);
