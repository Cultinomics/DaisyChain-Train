// A miniature transformer language model that trains THROUGH the verified INT8
// units: every matrix product in the forward pass — QKV projections, attention
// scores, attention·values, output projection, the MLP, and the unembedding —
// runs through the verified multiply LUT (an emulated INT8 tensor core).
// Backward is a straight-through estimator in float (the integer path has no
// gradient), exactly like the Python VerifiedLinear.
//
// Task: next-character prediction on a deterministic, self-generated corpus
// (every peer builds the same text from the same seed — nothing to download).
(function (root) {
  "use strict";

  let TC, V;   // TrainCore / Verified — resolved per environment at the end

  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function randn(n, rng) { const o = new Float32Array(n); for (let i = 0; i < n; i += 2) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); const m = Math.sqrt(-2 * Math.log(u)); o[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) o[i + 1] = m * Math.sin(2 * Math.PI * v); } return o; }

  // ---- corpus: deterministic cottagecore prose, identical on every peer -----
  const W_ADJ = ["mossy", "golden", "amber", "quiet", "little", "misty", "sunny", "wild", "cozy", "dusty", "merry", "brave"];
  const W_NOUN = ["fox", "hare", "owl", "badger", "toad", "sparrow", "otter", "deer", "mushroom", "acorn", "willow", "robin", "river", "meadow", "garden", "lantern"];
  const W_VERB = ["naps", "sings", "wanders", "hides", "dreams", "waits", "dances", "listens", "rests", "grows"];
  const W_PREP = ["by", "under", "near", "beside", "beyond", "inside"];
  function buildCorpus() {
    const rng = mulberry32(20260712);
    const pick = (a) => a[Math.floor(rng() * a.length)];
    let s = "";
    while (s.length < 60000)
      s += `the ${pick(W_ADJ)} ${pick(W_NOUN)} ${pick(W_VERB)} ${pick(W_PREP)} the ${pick(W_ADJ)} ${pick(W_NOUN)}. `;
    return s;
  }
  // ---- tokenizer -------------------------------------------------------------
  // Spikewhale tokenizer (tokenizer.json): byte-level greedy longest-match
  // ("length-max"), ~16.5k tokens. Until it loads (or if the file is missing)
  // a 96-char byte-level vocab keeps the app working — but ALL devices in a
  // group must use the same tokenizer (the config broadcast enforces it).
  const FALLBACK_CHARS = [...Array(95)].map((_, i) => String.fromCharCode(32 + i)).concat(["\n"]);
  let tok = {
    name: "char-96 (fallback)",
    vocab: Object.fromEntries(FALLBACK_CHARS.map((c, i) => [c, i])),
    ids: FALLBACK_CHARS, maxLen: 1, size: FALLBACK_CHARS.length,
    unk: 0, specials: new Set(),
  };
  tok.unk = tok.vocab[" "];
  function vocabSize() { return tok.size; }
  function tokenizerName() { return tok.name; }
  function loadTokenizerData(d) {              // plain {vocab, vocab_size, max_token_len}
    const ids = new Array(d.vocab_size);
    for (const [t, i] of Object.entries(d.vocab)) ids[i] = t;
    tok = { name: `Spikewhale length-max (${d.vocab_size} tokens)`,
            vocab: d.vocab, ids, maxLen: d.max_token_len || 24, size: d.vocab_size,
            unk: d.vocab["<unk>"] ?? 1,
            specials: new Set(["<pad>", "<unk>", "<bos>", "<eos>", ...(d.special_tokens || [])]) };
    IDS = encode(CORPUS);                      // re-tokenize whatever corpus is loaded
    return tok.name;
  }
  async function loadTokenizer(url) {
    const r = await fetch(url || "tokenizer.json");
    if (!r.ok) throw new Error(`tokenizer.json HTTP ${r.status}`);
    return loadTokenizerData(await r.json());
  }
  function toLatin1(s) { const b = new TextEncoder().encode(s); let o = ""; for (const x of b) o += String.fromCharCode(x); return o; }
  function encode(text) {                      // greedy longest match over bytes
    const s = toLatin1(text), out = [];
    let i = 0;
    while (i < s.length) {
      let m = null;
      for (let L = Math.min(tok.maxLen, s.length - i); L > 0; L--) {
        const sub = s.substr(i, L);
        if (sub in tok.vocab) { m = sub; break; }
      }
      if (m === null) { out.push(tok.unk); i++; continue; }
      out.push(tok.vocab[m]); i += m.length;
    }
    return Int32Array.from(out);
  }
  function decode(idArr) {
    let s = "";
    for (const id of idArr) {
      const t = tok.ids[id];
      if (t === undefined || tok.specials.has(t)) continue;
      s += t;
    }
    const bytes = Uint8Array.from([...s].map(c => c.charCodeAt(0)));
    return new TextDecoder().decode(bytes);
  }
  let CORPUS = buildCorpus();
  let IDS = encode(CORPUS);
  let DATASET = "built-in corpus";

  // Stream real training text from any public HuggingFace dataset with a
  // `text` column, via the datasets-server rows API. Each device pulls its own
  // random slice (that's data parallelism — batches were always per-device
  // anyway). Offline or on API failure the built-in corpus stays in place.
  const DEFAULT_DS = "HuggingFaceFW/fineweb-edu";
  async function streamDataset(ds) {
    ds = (ds || DEFAULT_DS).trim();
    if (!/^[\w.-]+\/[\w.-]+$/.test(ds)) throw new Error(`invalid dataset id "${ds}"`);
    const offset = Math.floor(Math.random() * 9900);
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(ds)}` +
                `&config=default&split=train&offset=${offset}&length=100`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`datasets-server HTTP ${r.status}`);
    const j = await r.json();
    const text = j.rows.map(x => String(x.row.text ?? "")).join("\n").replace(/[^\x20-\x7e\n]/g, " ");
    if (text.length < 10000) throw new Error("too little text returned (does the dataset have a `text` column?)");
    CORPUS = text.slice(0, 500000);
    IDS = encode(CORPUS);
    DATASET = `${ds} (streamed)`;
    return { name: DATASET, chars: CORPUS.length };
  }
  const streamFineWebEdu = () => streamDataset(DEFAULT_DS);
  function datasetName() { return DATASET; }

  // ---- verified matmul: block-scaled INT8 through the units ------------------
  // CUTLASS ex. 67/81 blockwise scaling: per-row activation scales × per-column
  // weight scales, one exact LUT/DP4A GEMM, dequant (+ optional fused ReLU) in
  // the kernel epilogue (ex. 12). Replaces the per-tensor 3-pass: same outlier
  // robustness at one third of the unit ops.
  async function vmm(Xf, Wf, m, k, n, ctx, relu) {
    // ctx.audit re-checks random cells of this LIVE GEMM against the units
    return V.vgemmBlock(Xf, Wf, { m, k, n, batch: 1, relu: !!relu }, ctx.L, ctx.bgemm, ctx.audit);
  }

  // ---- layernorm (no affine) -------------------------------------------------
  function lnFwd(x, rows, C) {
    const y = new Float32Array(rows * C), sig = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let mu = 0; for (let j = 0; j < C; j++) mu += x[r * C + j]; mu /= C;
      let v = 0; for (let j = 0; j < C; j++) { const d = x[r * C + j] - mu; v += d * d; }
      const s = Math.sqrt(v / C + 1e-5); sig[r] = s;
      for (let j = 0; j < C; j++) y[r * C + j] = (x[r * C + j] - mu) / s;
    }
    return { y, sig };
  }
  function lnBwd(dy, y, sig, rows, C) {
    const dx = new Float32Array(rows * C);
    for (let r = 0; r < rows; r++) {
      let mdy = 0, mdyy = 0;
      for (let j = 0; j < C; j++) { mdy += dy[r * C + j]; mdyy += dy[r * C + j] * y[r * C + j]; }
      mdy /= C; mdyy /= C;
      for (let j = 0; j < C; j++) dx[r * C + j] = (dy[r * C + j] - mdy - y[r * C + j] * mdyy) / sig[r];
    }
    return dx;
  }

  // ---- model -----------------------------------------------------------------
  // cfg: { c: width, t: seq len, b: batch/device, layers, heads, steps, lr }
  // engine: the Compute backend object ({bgemm} for the fused WebGPU path) or a
  // legacy matmulInt8 function (Node tests, inference kit) -> CPU LUT mirror
  function init(cfg, L, engine, audit) {
    const c = cfg.c, layers = cfg.layers || 2, heads = cfg.heads || 2, hidden = 2 * c;
    let seed = 100;
    const mk = (nEl, scale) => { const w = randn(nEl, mulberry32(seed++)); for (let i = 0; i < nEl; i++) w[i] *= scale; return w; };
    const params = [], names = [];
    const add = (name, w) => { params.push(w); names.push(name); return w; };
    const m = {
      cfg: { ...cfg, layers, heads, hidden, vocab: vocabSize() },
      ctx: { L, bgemm: (engine && engine.bgemm) || null,
             att: (engine && engine.att) || null, fgemm: (engine && engine.fgemm) || null,
             audit: audit || null },
      emb: add("emb", mk(vocabSize() * c, 0.08)),
      pos: add("pos", mk(cfg.t * c, 0.02)),
      blocks: [], params, names,
    };
    for (let l = 0; l < layers; l++)
      m.blocks.push({
        Wq: add(`b${l}.Wq`, mk(c * c, 0.08)), Wk: add(`b${l}.Wk`, mk(c * c, 0.08)),
        Wv: add(`b${l}.Wv`, mk(c * c, 0.08)), Wo: add(`b${l}.Wo`, mk(c * c, 0.08)),
        W1: add(`b${l}.W1`, mk(c * hidden, 0.08)), W2: add(`b${l}.W2`, mk(hidden * c, 0.08)),
      });
    // weight-tied unembedding: logits use embᵀ (no separate Wu). Halves the
    // vocab-sized parameters — and with a 16k vocab that's ~half of ALL
    // parameters, so gradients over the wire shrink ~2× too.
    m.nParams = params.reduce((a, p) => a + p.length, 0);
    return m;
  }

  function sampleBatch(cfg) {
    const { b, t } = cfg;
    const X = new Int32Array(b * t), Y = new Int32Array(b * t);
    for (let i = 0; i < b; i++) {
      const off = Math.floor(Math.random() * (IDS.length - t - 1));
      for (let j = 0; j < t; j++) { X[i * t + j] = IDS[off + j]; Y[i * t + j] = IDS[off + j + 1]; }
    }
    return { X, Y };
  }

  // per-head slice helpers (q: BT×C row-major; head h occupies cols [h*hd, (h+1)*hd))
  function headSlice(q, bIdx, h, T, C, hd) {
    const out = new Float32Array(T * hd);
    for (let ti = 0; ti < T; ti++)
      for (let j = 0; j < hd; j++) out[ti * hd + j] = q[(bIdx * T + ti) * C + h * hd + j];
    return out;
  }
  function headUnslice(dst, src, bIdx, h, T, C, hd, accumulate) {
    for (let ti = 0; ti < T; ti++)
      for (let j = 0; j < hd; j++) {
        const di = (bIdx * T + ti) * C + h * hd + j;
        dst[di] = (accumulate ? dst[di] : 0) + src[ti * hd + j];
      }
  }

  // ---- forward THROUGH the verified units (caches kept for STE backward) -----
  async function forward(m, X, Y) {
    const { c: C, t: T, b: B, layers, heads, hidden, vocab } = m.cfg;
    const BT = B * T, hd = C / heads, ctx = m.ctx;
    const cache = { X, Y, blocks: [] };
    let x = new Float32Array(BT * C);
    for (let i = 0; i < BT; i++) {
      const id = X[i], tpos = i % T;
      for (let j = 0; j < C; j++) x[i * C + j] = m.emb[id * C + j] + m.pos[tpos * C + j];
    }
    for (let l = 0; l < layers; l++) {
      const bl = m.blocks[l], cb = { xin: x };
      const l1 = lnFwd(x, BT, C); cb.ln1 = l1;
      const q = await vmm(l1.y, bl.Wq, BT, C, C, ctx);
      const k = await vmm(l1.y, bl.Wk, BT, C, C, ctx);
      const v = await vmm(l1.y, bl.Wv, BT, C, C, ctx);
      cb.q = q; cb.k = k; cb.v = v;
      const scale = 1 / Math.sqrt(hd);
      // gather-FUSED attention (CUTLASS ex. 36/52): the kernels read q/k/v in
      // their natural BT×C layout with head-strided indexing and scatter ctx
      // straight back — no JS gather copies, no kᵀ transpose. All B×H heads in
      // one dispatch per stage, every product through the verified units.
      const BH = B * heads, dAtt = { B, T, heads, hd };
      // per-(token,head) row quantization: the (BT·heads)×hd view IS the buffer
      const qq = V.quantizeRows(q, BT * heads, hd), kq = V.quantizeRows(k, BT * heads, hd);
      const sAll = ctx.att ? await ctx.att.scores(qq.q, kq.q, qq.s, kq.s, dAtt)
                           : V.attScoresJS(qq.q, kq.q, qq.s, kq.s, dAtt, ctx.L);
      const aAll = new Float32Array(BH * T * T);                   // causal softmax
      for (let bz = 0; bz < BH; bz++) {
        const so = bz * T * T;
        for (let ti = 0; ti < T; ti++) {
          let mx = -1e30;
          for (let tj = 0; tj <= ti; tj++) mx = Math.max(mx, sAll[so + ti * T + tj] * scale);
          let z = 0;
          for (let tj = 0; tj <= ti; tj++) { const e = Math.exp(sAll[so + ti * T + tj] * scale - mx); aAll[so + ti * T + tj] = e; z += e; }
          for (let tj = 0; tj <= ti; tj++) aAll[so + ti * T + tj] /= z;
        }
      }
      const aq = V.quantizeRows(aAll, BH * T, T);
      const vq = V.quantizeHeadCols(v, B, T, heads, hd);
      const ctxOut = ctx.att ? await ctx.att.ctx(aq.q, vq.q, aq.s, vq.s, dAtt)
                             : V.attCtxJS(aq.q, vq.q, aq.s, vq.s, dAtt, ctx.L);
      cb.aAll = aAll;                                  // backward slices heads from q/k/v/aAll
      cb.ctxOut = ctxOut;
      const attnOut = await vmm(ctxOut, bl.Wo, BT, C, C, ctx);
      const x2 = new Float32Array(BT * C);
      for (let i = 0; i < x2.length; i++) x2[i] = x[i] + attnOut[i];
      cb.x2 = x2;
      const l2 = lnFwd(x2, BT, C); cb.ln2 = l2;
      const h1 = await vmm(l2.y, bl.W1, BT, C, hidden, ctx, true);  // ReLU fused in the epilogue
      const mask = new Uint8Array(h1.length);
      for (let i = 0; i < h1.length; i++) if (h1[i] > 0) mask[i] = 1;
      cb.h1 = h1; cb.mask = mask;
      const mlpOut = await vmm(h1, bl.W2, BT, hidden, C, ctx);
      x = new Float32Array(BT * C);
      for (let i = 0; i < x.length; i++) x[i] = x2[i] + mlpOut[i];
      cache.blocks.push(cb);
    }
    const lf = lnFwd(x, BT, C); cache.lnf = lf; cache.xf = x;
    const logits = await vmm(lf.y, TC.transpose(m.emb, vocab, C), BT, C, vocab, ctx);  // tied: embᵀ
    // cross-entropy + dlogits
    let loss = 0;
    const dlogits = new Float32Array(BT * vocab);
    for (let i = 0; i < BT; i++) {
      let mx = -1e30;
      for (let j = 0; j < vocab; j++) mx = Math.max(mx, logits[i * vocab + j]);
      let z = 0;
      for (let j = 0; j < vocab; j++) z += Math.exp(logits[i * vocab + j] - mx);
      const lz = Math.log(z) + mx;
      loss += lz - logits[i * vocab + Y[i]];
      for (let j = 0; j < vocab; j++)
        dlogits[i * vocab + j] = (Math.exp(logits[i * vocab + j] - lz) - (j === Y[i] ? 1 : 0)) / BT;
    }
    loss /= BT;
    cache.dlogits = dlogits;
    return { loss, cache, logits };
  }

  // ---- STE backward (float), mirrors forward exactly --------------------------
  // The two vocab-sized matmuls run on the split-K f32 GPU kernel when
  // available (CUTLASS ex. 06) — same float math, off the JS thread.
  async function backward(m, cache) {
    const { c: C, t: T, b: B, layers, heads, hidden, vocab } = m.cfg;
    const BT = B * T, hd = C / heads, mm = TC.matmul, tr = TC.transpose;
    const g = m.params.map(p => new Float32Array(p.length));
    const gi = Object.fromEntries(m.names.map((n, i) => [n, i]));
    // tied unembed: logits = lnf @ embᵀ, so the unembedding gradient flows
    // straight into emb — dlogitsᵀ @ lnf is V×C, emb's own shape
    let dlnfIn;
    if (m.ctx.fgemm) {
      [g[gi.emb], dlnfIn] = await Promise.all([
        m.ctx.fgemm(cache.dlogits, cache.lnf.y, { m: vocab, k: BT, n: C, transA: true }),
        m.ctx.fgemm(cache.dlogits, m.emb, { m: BT, k: vocab, n: C }),      // split-K shape
      ]);
    } else {
      g[gi.emb] = mm(tr(cache.dlogits, BT, vocab), cache.lnf.y, vocab, BT, C);
      dlnfIn = mm(cache.dlogits, m.emb, BT, vocab, C);
    }
    let dx = lnBwd(dlnfIn, cache.lnf.y, cache.lnf.sig, BT, C);
    const scale = 1 / Math.sqrt(hd);
    for (let l = layers - 1; l >= 0; l--) {
      const bl = m.blocks[l], cb = cache.blocks[l];
      // mlp: x3 = x2 + relu(ln2 @ W1) @ W2
      const dmlpOut = dx;                              // residual passthrough handled below
      g[gi[`b${l}.W2`]] = mm(tr(cb.h1, BT, hidden), dmlpOut, hidden, BT, C);
      const dh1 = mm(dmlpOut, tr(bl.W2, hidden, C), BT, C, hidden);
      for (let i = 0; i < dh1.length; i++) if (!cb.mask[i]) dh1[i] = 0;
      g[gi[`b${l}.W1`]] = mm(tr(cb.ln2.y, BT, C), dh1, C, BT, hidden);
      const dln2in = lnBwd(mm(dh1, tr(bl.W1, C, hidden), BT, hidden, C), cb.ln2.y, cb.ln2.sig, BT, C);
      const dx2 = new Float32Array(BT * C);
      for (let i = 0; i < dx2.length; i++) dx2[i] = dx[i] + dln2in[i];
      // attention: x2 = xin + (ctxOut @ Wo)
      g[gi[`b${l}.Wo`]] = mm(tr(cb.ctxOut, BT, C), dx2, C, BT, C);
      const dctx = mm(dx2, tr(bl.Wo, C, C), BT, C, C);
      const dq = new Float32Array(BT * C), dk = new Float32Array(BT * C), dv = new Float32Array(BT * C);
      for (let bi = 0; bi < B; bi++) for (let h = 0; h < heads; h++) {
        const bz = bi * heads + h;
        const a = cb.aAll.subarray(bz * T * T, (bz + 1) * T * T);
        const qh = headSlice(cb.q, bi, h, T, C, hd);
        const kh = headSlice(cb.k, bi, h, T, C, hd);
        const vh = headSlice(cb.v, bi, h, T, C, hd);
        const dch = headSlice(dctx, bi, h, T, C, hd);        // T×hd
        const dvh = mm(tr(a, T, T), dch, T, T, hd);          // aᵀ @ dctx
        const da = mm(dch, tr(vh, T, hd), T, hd, T);         // dctx @ vᵀ
        const ds = new Float32Array(T * T);                  // softmax bwd (causal)
        for (let ti = 0; ti < T; ti++) {
          let dot = 0;
          for (let tj = 0; tj <= ti; tj++) dot += da[ti * T + tj] * a[ti * T + tj];
          for (let tj = 0; tj <= ti; tj++) ds[ti * T + tj] = a[ti * T + tj] * (da[ti * T + tj] - dot) * scale;
        }
        const dqh = mm(ds, kh, T, T, hd);                    // ds @ k
        const dkh = mm(tr(ds, T, T), qh, T, T, hd);          // dsᵀ @ q
        headUnslice(dq, dqh, bi, h, T, C, hd, true);
        headUnslice(dk, dkh, bi, h, T, C, hd, true);
        headUnslice(dv, dvh, bi, h, T, C, hd, true);
      }
      g[gi[`b${l}.Wq`]] = mm(tr(cb.ln1.y, BT, C), dq, C, BT, C);
      g[gi[`b${l}.Wk`]] = mm(tr(cb.ln1.y, BT, C), dk, C, BT, C);
      g[gi[`b${l}.Wv`]] = mm(tr(cb.ln1.y, BT, C), dv, C, BT, C);
      const dln1in = new Float32Array(BT * C);
      const addIn = (dsrc, W) => { const t2 = mm(dsrc, tr(W, C, C), BT, C, C); for (let i = 0; i < dln1in.length; i++) dln1in[i] += t2[i]; };
      addIn(dq, bl.Wq); addIn(dk, bl.Wk); addIn(dv, bl.Wv);
      const dxin = lnBwd(dln1in, cb.ln1.y, cb.ln1.sig, BT, C);
      dx = new Float32Array(BT * C);
      for (let i = 0; i < dx.length; i++) dx[i] = dx2[i] + dxin[i];
    }
    // embedding + positional
    const ge = g[gi.emb], gp = g[gi.pos];
    for (let i = 0; i < BT; i++) {
      const id = cache.X[i], tpos = i % T;
      for (let j = 0; j < C; j++) { ge[id * C + j] += dx[i * C + j]; gp[tpos * C + j] += dx[i * C + j]; }
    }
    // flatten
    const flat = new Float32Array(m.nParams);
    let off = 0;
    for (const t of g) { flat.set(t, off); off += t.length; }
    return flat;
  }

  async function trainStep(m) {
    const { X, Y } = sampleBatch(m.cfg);
    const { loss, cache } = await forward(m, X, Y);
    const grad = await backward(m, cache);
    return { loss, grad };
  }

  function applyUpdate(m, upd) {          // W -= upd (lr folded in by the optimizer)
    let off = 0;
    for (const p of m.params) { for (let i = 0; i < p.length; i++) p[i] -= upd[off + i]; off += p.length; }
  }
  function getFlatParams(m) {
    const flat = new Float32Array(m.nParams);
    let off = 0;
    for (const p of m.params) { flat.set(p, off); off += p.length; }
    return flat;
  }
  function setFlatParams(m, flat) {
    let off = 0;
    for (const p of m.params) { p.set(flat.subarray(off, off + p.length)); off += p.length; }
  }

  // greedy sampling — watch the model actually speak
  async function generate(m, prompt, nChars) {
    const { t: T } = m.cfg;
    let ids = [...encode(prompt)];
    for (let step = 0; step < nChars; step++) {
      const win = ids.slice(-T);
      const X = new Int32Array(T), Y = new Int32Array(T);
      for (let i = 0; i < win.length; i++) X[T - win.length + i] = win[i];
      const save = m.cfg.b; m.cfg.b = 1;
      const { logits } = await forward(m, X, Y);
      m.cfg.b = save;
      const row = (T - 1) * m.cfg.vocab;
      let best = 0, bv = -1e30;
      for (let j = 0; j < m.cfg.vocab; j++) if (logits[row + j] > bv) { bv = logits[row + j]; best = j; }
      ids.push(best);
    }
    return decode(ids);
  }

  const api = { init, trainStep, applyUpdate, getFlatParams, setFlatParams, generate,
                streamFineWebEdu, streamDataset, datasetName, loadTokenizer, loadTokenizerData,
                vocabSize, tokenizerName, encode, decode };
  if (typeof module !== "undefined" && module.exports) { TC = require("./traincore.js"); V = require("./verified_core.js"); module.exports = api; }
  else { TC = root.TrainCore; V = root.Verified; root.Transformer = api; }
})(typeof self !== "undefined" ? self : this);
