// Run: node --test tests/kimi-k31-regression.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const ctx = vm.createContext({});
for (const file of ['js/data.js', 'js/calc.js', 'js/calc_weight.js', 'js/calc_deploy.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), ctx, { filename: file });
}
const model = vm.runInContext("MODEL_DATA.models.find(m => m.id === 'kimi-k3.1-pro')", ctx);
const k3 = vm.runInContext("MODEL_DATA.models.find(m => m.id === 'kimi-k3')", ctx);
test('official K3.1 layer partition and multimodal metadata', () => {
  assert.ok(model);
  assert.equal(model.formula, 'kda_gated_dsa');
  assert.equal(model.fields.kda_layers.length, 69);
  assert.equal(model.fields.dsa_layers.length, 24);
  assert.equal(new Set([...model.fields.kda_layers, ...model.fields.dsa_layers]).size, 93);
  assert.equal(model.fields.num_nextn_predict_layers, 0);
  assert.equal(model.vision_fields.params, 401000000);
  assert.equal(model.max_position_embeddings, 1048576);
});
test('DSA indexer is stored once per sparse layer, not per query head', () => {
  const tokens = 4096, prec = 2, idxPrec = 1;
  const r = ctx.calcKvCache(model, tokens, prec, idxPrec, { includeLinear: true });
  const base = ctx.calcKvCache(k3, tokens, prec, idxPrec, { includeLinear: true });
  const expected = 24 * 128 * tokens * idxPrec;
  assert.equal(r.idxBytes, expected);
  assert.equal(r.kvBytes, base.kvBytes);
  assert.equal(r.kvBytes + r.idxBytes, base.kvBytes + expected);
  assert.equal(r.idxLayers, 24);
  assert.equal(r.idxLayerBytes.filter(x => x > 0).length, 24);
  assert.equal(r.idxLayerBytes.reduce((a,b)=>a+b,0), expected);
  assert.equal(ctx.modelHasIndexer(model), true);
});
test('KDA state toggle does not affect DSA indexer; KV and indexer scale with precision', () => {
  const a = ctx.calcKvCache(model, 1024, 2, 2, { includeLinear: false });
  const b = ctx.calcKvCache(model, 2048, 1, 1, { includeLinear: false });
  assert.equal(a.kvBytes, b.kvBytes);
  assert.equal(a.idxBytes, b.idxBytes);
  assert.equal(a.idxBytes, 24 * 128 * 1024 * 2);
});
