/* MoE Communication Simulation Engine
 * Pure math, no DOM access. Global MoEEngine namespace.
 * ES5 style: var, function, no modules.
 *
 * Routing model:
 *   M = N * E experts. Expert j resides on GPU floor(j / E).
 *   Each token (round-robin source GPU) picks top-k experts by affinity.
 *   Affinity = popularity_weight + noise.
 *   Popularity uses Zipf-like distribution controlled by skew (0=uniform, 1=extreme).
 *   dispatchMatrix counts every (token, expert) pair — one entry per expert assignment.
 *   expertFlow[src][dst][e] tracks per-expert flow for detailed tooltips.
 *   expertLoads tracks per-expert counts (same total as dispatchMatrix + expertFlow).
 */


function _mulberry32(seed) {
  var a = seed;
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    var t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}


function _shuffle(arr, rng) {
  var i, j, tmp;
  for (i = arr.length - 1; i > 0; i--) {
    j = Math.floor(rng() * (i + 1));
    tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

var MoEEngine = {

  /* Main routing function.
   * params: { numGpus, expertsPerGpu, numTokens, topk, skew, seed }
   * Returns routing result with matrices and metrics.
   */
  route: function(params) {
    var N = params.numGpus;
    var E = params.expertsPerGpu;
    var T = params.numTokens;
    var k = params.topk;
    var skew = params.skew;
    var seed = params.seed;

    var M = N * E;
    var rng = _mulberry32(seed);
    var srcRng = _mulberry32(seed ^ 0x9E3779B9);

    /* --- Step 1: Expert popularity weights (Zipf-like, skew-controlled) --- */
    var perm = [];
    var i;
    for (i = 0; i < M; i++) perm[i] = i;
    _shuffle(perm, rng);

    var hotness = [];
    for (i = 0; i < M; i++) hotness[i] = 0;
    for (i = 0; i < M; i++) {
      var rank = i + 1;
      var expertIdx = perm[i];
      hotness[expertIdx] = 1.0 / rank;
    }

    var weights = [];
    var wMax = 0, wMin = Infinity;
    for (i = 0; i < M; i++) {
      weights[i] = 1.0 + skew * hotness[i];
      if (weights[i] > wMax) wMax = weights[i];
      if (weights[i] < wMin) wMin = weights[i];
    }
    var noiseAmp = (wMax - wMin) + 0.5;

    /* --- Step 2: Token routing --- */
    var dispatchMatrix = [];
    var expertLoads = [];
    var expertFlow = [];
    var tokenCounts = [];
    var g, e;
    for (g = 0; g < N; g++) {
      dispatchMatrix[g] = [];
      expertLoads[g] = [];
      expertFlow[g] = [];
      tokenCounts[g] = 0;
      for (e = 0; e < E; e++) {
        expertLoads[g][e] = 0;
      }
      for (var dst = 0; dst < N; dst++) {
        dispatchMatrix[g][dst] = 0;
        expertFlow[g][dst] = [];
        for (e = 0; e < E; e++) expertFlow[g][dst][e] = 0;
      }
    }

    var t;
    for (t = 0; t < T; t++) {
      var srcGpu = Math.floor(srcRng() * N);
      tokenCounts[srcGpu]++;

      var affinities = [];
      for (i = 0; i < M; i++) {
        affinities[i] = { idx: i, val: weights[i] + rng() * noiseAmp };
      }

      affinities.sort(function(a, b) { return b.val - a.val; });

      var selected = [];
      var selCount = Math.min(k, M);
      for (i = 0; i < selCount; i++) {
        selected[i] = affinities[i].idx;
      }

      var gpuSeen = [];
      for (g = 0; g < N; g++) gpuSeen[g] = false;

      for (i = 0; i < selected.length; i++) {
        var expertIdx = selected[i];
        var expertGpu = Math.floor(expertIdx / E);
        var localExpert = expertIdx % E;

        expertLoads[expertGpu][localExpert]++;
        expertFlow[srcGpu][expertGpu][localExpert]++;

        if (!gpuSeen[expertGpu]) {
          dispatchMatrix[srcGpu][expertGpu]++;
          gpuSeen[expertGpu] = true;
        }
      }
    }

    /* --- Step 3: Build combine matrix (transpose of dispatch) --- */
    var combineMatrix = [];
    for (g = 0; g < N; g++) {
      combineMatrix[g] = [];
      for (var s = 0; s < N; s++) {
        combineMatrix[g][s] = dispatchMatrix[s][g];
      }
    }

    /* --- Step 4: Compute metrics --- */
    /* Self-loop (g==d) entries are excluded from all communication
     * metrics: a token routed to an expert on its source GPU needs
     * no network transfer.  expertLoads still counts every assignment. */
    var totalTokensMoved = 0;
    var totalExpertAssignments = 0;
    var maxExpertLoad = 0;
    var minExpertLoad = Infinity;
    var maxGpuRecv = 0;
    var maxGpuSend = 0;
    var hotGpuIdx = 0;
    var hotExpertGpu = 0;
    var hotExpertLocal = 0;
    var gpuSends = [];

    for (g = 0; g < N; g++) {
      var gpuSend = 0;
      var gpuRecv = 0;
      for (var d = 0; d < N; d++) {
        if (g === d) continue;
        totalTokensMoved += dispatchMatrix[g][d];
        gpuSend += dispatchMatrix[g][d];
        gpuRecv += dispatchMatrix[d][g];
      }
      gpuSends[g] = gpuSend;
      if (gpuSend > maxGpuSend) maxGpuSend = gpuSend;
      if (gpuRecv > maxGpuRecv) {
        maxGpuRecv = gpuRecv;
        hotGpuIdx = g;
      }

      for (e = 0; e < E; e++) {
        var load = expertLoads[g][e];
        totalExpertAssignments += load;
        if (load > maxExpertLoad) {
          maxExpertLoad = load;
          hotExpertGpu = g;
          hotExpertLocal = e;
        }
        if (load < minExpertLoad) minExpertLoad = load;
      }
    }

    if (minExpertLoad === Infinity) minExpertLoad = 0;

    var avgExpertLoad = totalExpertAssignments / M;

    var imbalanceRatio = avgExpertLoad > 0 ? maxExpertLoad / avgExpertLoad : 1.0;

    return {
      numGpus: N,
      expertsPerGpu: E,
      numTokens: T,
      topk: k,
      skew: skew,
      dispatchMatrix: dispatchMatrix,
      expertLoads: expertLoads,
      expertFlow: expertFlow,
      tokenCounts: tokenCounts,
      combineMatrix: combineMatrix,
      popularityWeights: weights,
      totalTokensMoved: totalTokensMoved,
      totalTokensRoundTrip: 2 * totalTokensMoved,
      maxExpertLoad: maxExpertLoad,
      minExpertLoad: minExpertLoad,
      avgExpertLoad: avgExpertLoad,
      imbalanceRatio: imbalanceRatio,
      maxGpuRecvLoad: maxGpuRecv,
      maxGpuSendLoad: maxGpuSend,
      hotExpert: { gpu: hotExpertGpu, expert: hotExpertLocal, load: maxExpertLoad },
      hotGpu: { gpu: hotGpuIdx, recvLoad: maxGpuRecv, sendLoad: gpuSends[hotGpuIdx] }
    };
  },

  /* Estimate communication time (seconds)
   * tokensMoved: number of tokens transferred
   * tokenSizeBytes: bytes per token
   * bandwidthGBs: bandwidth in GB/s
   */
  estimateCommTime: function(tokensMoved, tokenSizeBytes, bandwidthGBs) {
    if (bandwidthGBs <= 0) return 0;
    var bytes = tokensMoved * tokenSizeBytes;
    var bytesPerSec = bandwidthGBs * 1e9;
    return bytes / bytesPerSec;
  },

  /* Estimate compute time (seconds)
   * maxExpertLoad: tokens on the busiest expert (bottleneck)
   * expertSpeedTokensPerSec: processing speed per expert
   */
  estimateComputeTime: function(maxExpertLoad, expertSpeedTokensPerSec) {
    if (expertSpeedTokensPerSec <= 0) return 0;
    return maxExpertLoad / expertSpeedTokensPerSec;
  },

  
  generateSeed: function() {
    return Math.floor(Math.random() * 4294967296);
  }
};
