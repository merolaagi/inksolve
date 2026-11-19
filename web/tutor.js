/* Code, step by step: a Python-Tutor-style stepper over one real training step.
   The server runs server/trace_code.py under sys.settrace and returns every line's state. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (n) => Math.round(n).toLocaleString();
  const num = (v) => (v == null ? "–" : v === 0 ? "0" : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? String(+v.toPrecision(4)) : v.toExponential(2));
  const shp = (s) => (s.length ? s.join("×") : "scalar");
  const T = { data: null, k: 0, params: [], hl: null };

  async function api(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
    return j;
  }

  // ---------------------------------------------------------------- heatmaps
  const cache = new Map();
  function colours() {
    const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    return { neg: hex(dark ? "#3987e5" : "#2a78d6"), pos: hex(dark ? "#e8743f" : "#eb6834"), mid: hex(dark ? "#2c2c30" : "#f1efea"), empty: hex(dark ? "#161616" : "#ffffff") };
  }
  function heatCanvas(b64, hw, key) {
    const k = key + (document.documentElement.dataset.theme || "");
    if (cache.has(k)) return cache.get(k);
    const [H, W] = hw, s = atob(b64), c = document.createElement("canvas");
    c.width = W; c.height = H;
    const g = c.getContext("2d"), img = g.createImageData(W, H), P = colours();
    for (let i = 0; i < H * W; i++) {
      const q = s.charCodeAt(i);
      let col;
      if (q === 0) col = P.empty;
      else { const t = (q - 128) / 127, a = Math.min(1, Math.abs(t)) ** 0.55, e = t < 0 ? P.neg : P.pos; col = [0, 1, 2].map((j) => Math.round(P.mid[j] + (e[j] - P.mid[j]) * a)); }
      img.data[i * 4] = col[0]; img.data[i * 4 + 1] = col[1]; img.data[i * 4 + 2] = col[2]; img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    cache.set(k, c);
    return c;
  }
  function drawInto(canvas, src, maxW, maxH) {
    const s = Math.max(0.25, Math.min(maxW / src.width, maxH / src.height, 8));
    canvas.width = Math.max(1, Math.round(src.width * s)); canvas.height = Math.max(1, Math.round(src.height * s));
    const g = canvas.getContext("2d"); g.imageSmoothingEnabled = s < 1; g.drawImage(src, 0, 0, canvas.width, canvas.height);
  }
  const heatEl = (o, maxW = 300, maxH = 150, delta = false) => {
    const c = document.createElement("canvas");
    drawInto(c, heatCanvas(delta ? o.delta : o.heat, o.hw, o.id + (delta ? "d" : "")), maxW, maxH);
    c.title = delta ? `how much each number moved (largest ±${num(o.dmax)})` : `blue negative · orange positive · largest ±${num(o.max)}`;
    return c;
  };

  // ---------------------------------------------------------------- code listing
  const KW = /\b(def|return|for|in|with|import|from|as|if|else|min|sum)\b/g;
  function codeHtml(line) {
    const i = line.indexOf("#");
    const code = i >= 0 && !line.slice(0, i).includes('"') ? line.slice(0, i) : line, cm = code === line ? "" : line.slice(i);
    return esc(code).replace(KW, '<span class="kw">$1</span>') + (cm ? `<span class="cm">${esc(cm)}</span>` : "");
  }
  function renderCode() {
    const d = T.data;
    const start = d.code.findIndex((l) => l.startsWith("def forward"));
    $("#cd-lines").innerHTML = d.code.map((l, i) => i < start ? "" : `<div class="t-line" data-ln="${i + 1}"><span class="ar"></span><span class="ln">${i + 1}</span><span class="tx">${codeHtml(l) || " "}</span></div>`).join("");
    $$("#cd-lines .t-line").forEach((el) => (el.onclick = () => {
      const ln = +el.dataset.ln;
      let j = T.data.steps.findIndex((s, i) => i > T.k && s.line === ln);
      if (j < 0) j = T.data.steps.findIndex((s) => s.line === ln);
      if (j >= 0) go(j);
    }));
  }

  // ---------------------------------------------------------------- phases
  const PHASES = [
    ["Start", (s) => s.func === "training_step" && /zero_grad/.test(line(s))],
    ["Forward pass", (s) => s.func === "forward"],
    ["The classifier", (s) => s.func === "forward" && /flatten/.test(line(s))],
    ["Loss", (s) => /softmax/.test(line(s))],
    ["Backward pass", (s) => /backward/.test(line(s))],
    ["Gradient by hand", (s) => /error =/.test(line(s))],
    ["Update", (s) => /no_grad/.test(line(s))],
    ["conv1 weights move", (s) => /layer\.weight -=/.test(line(s))],
    ["fc1 weights move", (s, i) => /layer\.weight -=/.test(line(s)) && layerAt(i) === "fc1"],
    ["Done", (s) => s.event === "return" && s.func === "training_step"],
  ];
  const line = (s) => T.data.code[s.line - 1] || "";
  function layerAt(i) { const s = T.data.steps[i]; const f = s.frames[s.frames.length - 1]; const v = f.vars.find((x) => x.name === "layer"); return v ? v.v.replace("net.", "") : null; }
  function renderPhases() {
    $("#cd-phases").innerHTML = PHASES.map(([n, f]) => { const i = T.data.steps.findIndex((s, j) => f(s, j)); return i < 0 ? "" : `<li><button data-go="${i}">${esc(n)} <span style="color:var(--ds-faint)">${i + 1}</span></button></li>`; }).join("");
    $$("#cd-phases button").forEach((b) => (b.onclick = () => go(+b.dataset.go)));
  }

  // ---------------------------------------------------------------- one step
  function paramsAt(k) {
    const p = {};
    for (let i = 0; i <= k; i++) Object.assign(p, T.data.steps[i].params);
    return p;
  }
  function go(k) {
    const d = T.data;
    T.k = Math.max(0, Math.min(d.steps.length - 1, k));
    const s = d.steps[T.k], prev = T.k > 0 ? d.steps[T.k - 1] : null;
    $("#cd-slider").value = T.k;
    $("#cd-count").textContent = `Step ${T.k + 1} of ${d.steps.length}`;
    $$("#cd-lines .t-line").forEach((el) => {
      const ln = +el.dataset.ln;
      const ran = prev && prev.line === ln, next = s.event !== "return" && s.line === ln;
      el.classList.toggle("ran", !!ran && !next);
      el.classList.toggle("next", next);
      el.querySelector(".ar").textContent = next ? "➜" : ran ? "➜" : "";
    });
    const nx = $("#cd-lines .t-line.next") || $("#cd-lines .t-line.ran");
    if (nx) { const box = $("#cd-lines"), r = nx.offsetTop - box.offsetTop; if (r < box.scrollTop + 30 || r > box.scrollTop + box.clientHeight - 50) box.scrollTop = r - box.clientHeight / 2; }
    $$("#cd-phases button").forEach((b) => b.setAttribute("aria-selected", "false"));
    const ph = $$("#cd-phases button").filter((b) => +b.dataset.go <= T.k).pop();
    if (ph) ph.setAttribute("aria-selected", "true");
    renderFrames(s, prev);
    renderObjects(s, prev);
    renderModel(T.k);
    $("#cd-note").innerHTML = note(prev, s);
    requestAnimationFrame(arrows);
  }

  const refsOf = (v) => (v.k === "ref" ? [v.ref] : v.k === "tuple" ? v.refs : []);
  function valueHtml(v) {
    if (v.k === "num") return `<span>${esc(typeof v.v === "number" ? num(v.v) : v.v)}</span>`;
    if (v.k === "ref") { const o = T.data.objects[v.ref]; return o.value != null && o.n === 1 ? `<span class="chip" data-ref="${v.ref}">tensor ${esc(num(o.value))}</span>` : `<span class="chip" data-ref="${v.ref}">tensor ${esc(shp(o.shape))}</span>`; }
    if (v.k === "tuple") return v.refs.map((r) => `<span class="chip" data-ref="${r}">${esc(shp(T.data.objects[r].shape))}</span>`).join(" ");
    if (v.k === "model") return `<span class="chip" data-model="1">the model</span>`;
    if (v.k === "layer") return `<span class="chip" data-layer="${esc(v.v.replace("net.", ""))}">${esc(v.v)}</span>`;
    return `<span>${esc(v.v)}</span>`;
  }
  function renderFrames(s, prev) {
    const prevVals = {};
    if (prev) prev.frames.forEach((f, fi) => f.vars.forEach((v) => (prevVals[fi + f.func + v.name] = JSON.stringify(v))));
    $("#cd-frames").innerHTML = s.frames.map((f, fi) => `<div class="t-frame ${fi === s.frames.length - 1 ? "active" : ""}"><b>${esc(f.func)}()</b>${f.vars.map((v) => {
      const fresh = prev && prevVals[fi + f.func + v.name] !== JSON.stringify(v);
      const lab = v.name === "label" && typeof v.v === "number" ? ` <span style="color:var(--ds-faint)">“${esc(T.data.classes[v.v])}”</span>` : "";
      return `<div class="t-var ${fresh ? "fresh" : ""}"><span class="nm">${esc(v.name)}</span><span class="vl">${valueHtml(v)}${lab}</span></div>`;
    }).join("") || '<div class="t-var"><span class="nm" style="color:var(--ds-faint)">no variables yet</span></div>'}</div>`).join("") + (s.event === "return" ? `<p class="ds-hint">${esc(s.func)}() is returning.</p>` : "");
  }
  function renderObjects(s, prev) {
    const seen = [], names = {};
    s.frames.slice().reverse().forEach((f) => f.vars.forEach((v) => refsOf(v).forEach((r) => { if (!seen.includes(r)) seen.push(r); names[r] = names[r] || []; if (!names[r].includes(v.name)) names[r].push(v.name); })));
    const prevRefs = new Set();
    if (prev) prev.frames.forEach((f) => f.vars.forEach((v) => refsOf(v).forEach((r) => prevRefs.add(r))));
    const host = $("#cd-objects");
    host.innerHTML = "";
    for (const r of seen) {
      const o = T.data.objects[r], isNew = prev && !prevRefs.has(r);
      const el = document.createElement("div");
      el.className = `t-obj ${isNew ? (o.prev ? "chg" : "new") : ""}`;
      el.dataset.obj = r;
      el.innerHTML = `<div class="hd"><b>${esc(names[r].join(", "))}</b><span>${esc(shp(o.shape))} · ${fmt(o.n)} number${o.n === 1 ? "" : "s"}</span>${isNew ? `<span class="tag">${o.prev ? "changed" : "new"}</span>` : ""}</div>`;
      if (o.n === 1) el.insertAdjacentHTML("beforeend", `<div class="big">${esc(num(o.value))}</div>`);
      else {
        el.appendChild(heatEl(o, 300, o.hw[0] === 1 ? 26 : 160));
        el.insertAdjacentHTML("beforeend", `<div class="st">min ${num(o.stats.min)} · max ${num(o.stats.max)} · mean ${num(o.stats.mean)}${o.stats.zeros > 0.01 ? ` · ${Math.round(o.stats.zeros * 100)}% zero` : ""}<br>[${o.preview.map(num).join(", ")}${o.n > o.preview.length ? ", …" : ""}]</div>`);
      }
      host.appendChild(el);
    }
    if (!seen.length) host.innerHTML = '<p class="ds-hint">No tensors in the frames yet.</p>';
  }
  const LAYERS = ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "conv4", "bn4", "conv5", "bn5", "fc1", "fc2"];
  function renderModel(k) {
    const p = paramsAt(k), pp = k > 0 ? paramsAt(k - 1) : {};
    const host = $("#cd-model");
    host.innerHTML = '<div class="t-prow" style="border-top:0;padding-top:0"><span></span><small>value (weight)</small><small>.grad</small></div>';
    for (const L of LAYERS) {
      const w = p[`${L}.weight`], b = p[`${L}.bias`];
      if (!w) continue;
      const ow = T.data.objects[w.w], og = w.g ? T.data.objects[w.g] : null;
      const chg = pp[`${L}.weight`] && (pp[`${L}.weight`].w !== w.w || pp[`${L}.bias`]?.w !== b.w);
      const gnew = pp[`${L}.weight`] && pp[`${L}.weight`].g !== w.g;
      const row = document.createElement("div");
      row.className = `t-prow ${chg ? "chg" : gnew ? "gnew" : ""}`;
      row.dataset.layer = L;
      row.innerHTML = `<b>${L}<br><small>${esc(shp(ow.shape))}</small></b>`;
      const c1 = document.createElement("div");
      c1.appendChild(heatEl(ow, 150, 72, false));
      if (chg && ow.delta) { c1.appendChild(heatEl(ow, 150, 72, true)); c1.insertAdjacentHTML("beforeend", `<small>moved: ${fmt(ow.changed)} numbers, at most ±${num(ow.maxmove)}</small>`); }
      const c2 = document.createElement("div");
      if (og) { c2.appendChild(heatEl(og, 150, 72)); c2.insertAdjacentHTML("beforeend", `<small>|grad| avg ${num(og.stats.abs)}</small>`); }
      else c2.innerHTML = '<span class="none">None</span>';
      row.append(c1, c2);
      host.appendChild(row);
    }
  }

  // ---------------------------------------------------------------- what the line just did
  function objOf(s, name) {
    for (const f of s.frames.slice().reverse()) { const v = f.vars.find((x) => x.name === name); if (v) return v.k === "ref" ? T.data.objects[v.ref] : v; }
    return null;
  }
  function note(prev, s) {
    const d = T.data;
    if (!prev) return `About to run <code>training_step()</code> with the mark, the label <b>“${esc(d.label)}”</b> and learning rate ${d.lr}. Press <b>Forward ▶</b> or →.`;
    const L = (d.code[prev.line - 1] || "").trim();
    const h = objOf(s, "h"), p = paramsAt(T.k);
    const conv = L.match(/net\.(conv\d)\.weight/);
    if (conv && L.startsWith("h = F.conv2d")) {
      const w = d.objects[p[`${conv[1]}.weight`].w], hw = h.shape[2] * h.shape[3];
      return `<b>${conv[1]}</b>: each of its ${w.shape[0]} filters (${w.shape[1]} × 3 × 3 = ${w.shape[1] * 9} weights) slides over all ${fmt(hw)} positions. <code>h</code> is now ${shp(h.shape)}: ${fmt(h.n)} numbers from ${fmt(w.n)} weights, ${fmt(w.n * hw)} multiply-adds.`;
    }
    if (L.startsWith("h = F.relu(normalize")) return `BatchNorm rescales each map with its learnt numbers, then ReLU turns every negative number into 0: <b>${Math.round(h.stats.zeros * 100)}%</b> of <code>h</code> is now zero. Zeros stop the signal, and later stop the gradient.`;
    if (L.startsWith("h = F.max_pool2d")) return `Each 2×2 square keeps only its largest number: <code>h</code> shrinks to ${shp(h.shape)} (${fmt(h.n)} numbers).`;
    if (L.startsWith("v = h.flatten")) return `The same ${fmt(objOf(s, "v").n)} numbers, laid out in one row so a matrix can multiply them.`;
    if (L.startsWith("a = F.relu(v @ net.fc1")) { const a = objOf(s, "a"); return `<b>fc1</b> is a 128 × 576 matrix: neuron <i>i</i> = row <i>i</i> of the weights times <code>v</code>, added up, plus a bias. That is ${fmt(128 * 576)} multiply-adds. ReLU then switches off ${Math.round(a.stats.zeros * 128)} of the 128 neurons.`; }
    if (L.startsWith("logits = a @ net.fc2")) return `<b>fc2</b>: 22 rows of 128 weights, each times <code>a</code>: one score per symbol (${fmt(22 * 128)} multiply-adds). The biggest score is the model's answer.`;
    if (L.startsWith("return logits")) return `<code>forward()</code> hands back <code>logits</code> and <code>a</code>; its frame closes.`;
    if (L.startsWith("logits, a = forward")) return `<code>forward()</code> ran: a new frame opened, did its work and closed. <code>logits</code> holds the 22 scores.`;
    if (L.startsWith("probs = F.softmax")) {
      const v = objOf(s, "probs").values, top = [...v.keys()].sort((x, y) => v[y] - v[x]).slice(0, 3);
      return `Softmax turns the scores into probabilities that add up to 1. Top three: ${top.map((i) => `“${esc(d.classes[i])}” ${(v[i] * 100).toFixed(1)}%`).join(", ")}. The answer it is being taught, “${esc(d.label)}”, has <b>${(v[d.classes.indexOf(d.label)] * 100).toFixed(2)}%</b>.`;
    }
    if (L.startsWith("loss = -torch.log")) { const lo = objOf(s, "loss"); return `<code>loss</code> = −log(probability of “${esc(d.label)}”) = <b>${num(lo.value)}</b>. 0 would mean certain and right; the bigger it is, the bigger every gradient will be.`; }
    if (L.startsWith("loss.backward")) { const n = Object.values(p).filter((x) => x.g).length; return `<b>Backpropagation.</b> PyTorch went back through every line of <code>forward()</code> in reverse, applying the chain rule, and gave all ${n} parameter tensors a <code>.grad</code> of the same shape. See the right-hand column of the model's weights: each one is now filled in.`; }
    if (L.startsWith("error =")) return `<code>error</code> = what the model said minus what it should have said (1 for “${esc(d.label)}”, 0 elsewhere): negative at the right answer, positive where it was too sure.`;
    if (L.startsWith("grad_fc2")) { const g = objOf(s, "grad_fc2"); return `<code>error</code> (22) times <code>a</code> (128), every pair: a 22 × 128 matrix, <b>exactly the numbers backward() put in <code>net.fc2.weight.grad</code></b>. Compare the two heatmaps. For the earlier layers the same rule is applied again and again, through each ReLU and convolution.`; }
    if (L.startsWith("with torch.no_grad")) return `From here on PyTorch stops recording: the update itself is not something to learn.`;
    if (L.startsWith("length =")) return `The length of all ${fmt(Object.values(p).reduce((a, x) => a + d.objects[x.w].n, 0))} gradient numbers together: <b>${num(objOf(s, "length").value)}</b>.`;
    if (L.startsWith("scale =")) { const sc = objOf(s, "scale"); return sc.v < 1 ? `Too long, so every gradient is shortened by <b>${num(sc.v)}</b>. One odd example can't throw the model far.` : `Short enough: <code>scale</code> = 1, gradients are used as they are.`; }
    if (L.startsWith("for layer in")) { const ly = layerAt(T.k); return ly ? `Next: <code>net.${ly}</code>.` : `Every layer has been updated.`; }
    const m = L.match(/^layer\.(weight|bias) -=/);
    if (m) {
      const ly = layerAt(T.k - 1) || layerAt(T.k), o = d.objects[p[`${ly}.${m[1]}`].w];
      return `<b>net.${ly}.${m[1]}</b> moved: new = old − ${d.lr} × scale × grad, for all ${fmt(o.n)} numbers at once. ${fmt(o.changed || 0)} of them changed; the largest move was ±${num(o.maxmove)}. In the model's weights, the top picture is the new value and the one under it shows the moves.`;
    }
    if (L.startsWith("net.zero_grad")) return `Every <code>.grad</code> is cleared (None), so this step's gradients don't add to an old step's.`;
    if (L.startsWith("return loss")) return `One training step done: ${fmt(Object.values(p).reduce((a, x) => a + d.objects[x.w].n, 0))} numbers each moved a little in the direction that makes “${esc(d.label)}” more likely next time.`;
    return "";
  }
  
  // ---------------------------------------------------------------- arrows from variables to objects
  function arrows() {
    const svg = $("#cd-arrows"), box = $("#tutor");
    if (!svg || !box.offsetParent || innerWidth < 640) { if (svg) svg.innerHTML = ""; return; }
    const B = box.getBoundingClientRect();
    const col = css("--ds-faint") || "#999";
    let h = `<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="${col}"/></marker></defs>`;
    $$("#cd-frames .chip[data-ref]").forEach((c) => {
      const o = $(`#cd-objects .t-obj[data-obj="${c.dataset.ref}"]`);
      if (!o) return;
      const a = c.getBoundingClientRect(), b = o.getBoundingClientRect();
      const x1 = a.right - B.left, y1 = a.top + a.height / 2 - B.top, x2 = b.left - B.left - 2, y2 = b.top + 14 - B.top;
      if (x2 < x1) return;
      const mx = (x1 + x2) / 2;
      h += `<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none" stroke="${col}" stroke-width="1.3" marker-end="url(#ah)"/>`;
    });
    svg.innerHTML = h;
  }
  addEventListener("resize", () => requestAnimationFrame(arrows));
  document.addEventListener("ds:theme", () => { cache.clear(); if (T.data) go(T.k); });

  // hover links between chips, objects and model rows
  document.addEventListener("mouseover", (e) => {
    const c = e.target.closest?.("#tutor .chip, #tutor .t-obj, #tutor .t-prow");
    $$("#tutor .hl").forEach((x) => x.classList.remove("hl"));
    if (!c) return;
    const ref = c.dataset.ref || c.dataset.obj, ly = c.dataset.layer;
    if (ref) $$(`#tutor [data-ref="${ref}"], #tutor [data-obj="${ref}"]`).forEach((x) => x.classList.add("hl"));
    if (ly) $$(`#tutor [data-layer="${ly}"]`).forEach((x) => x.classList.add("hl"));
  });

  // ---------------------------------------------------------------- controls
  $("#cd-first").onclick = () => go(0);
  $("#cd-prev").onclick = () => go(T.k - 1);
  $("#cd-next").onclick = () => go(T.k + 1);
  $("#cd-last").onclick = () => go(1e9);
  $("#cd-slider").oninput = (e) => go(+e.target.value);
  addEventListener("keydown", (e) => {
    if ($('[data-ds-view="code"]').hidden || !T.data || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) || document.querySelector(".ds-search.open")) return;
    const k = { ArrowRight: T.k + 1, ArrowLeft: T.k - 1, Home: 0, End: 1e9 }[e.key];
    if (k != null) { e.preventDefault(); go(k); }
  });
  const SAMPLES = ["7", "1", "b", "6", "2", "z", "x", "+"];
  let sample = { c: "7", seed: 3 };
  function fillSamples(classes) {
    $("#cd-samples").innerHTML = SAMPLES.map((c) => `<button data-s="${esc(c)}" class="${c === sample.c ? "on" : ""}" title="A synthetic ${esc(c)}">${esc(c)}</button>`).join("") + `<button data-s="" title="The mark from Inside the model">✎</button>`;
    $$("#cd-samples button").forEach((b) => (b.onclick = () => { sample = b.dataset.s ? { c: b.dataset.s, seed: Math.floor(Math.random() * 1e6) } : null; if (sample) $("#cd-label").value = sample.c; $$("#cd-samples button").forEach((x) => x.classList.toggle("on", x === b)); }));
    if (!$("#cd-label").options.length) { $("#cd-label").innerHTML = classes.map((c) => `<option>${esc(c)}</option>`).join(""); $("#cd-label").value = "1"; }
  }
  async function run() {
    const btn = $("#cd-run");
    btn.disabled = true; btn.textContent = "Running…";
    try {
      const body = { label: $("#cd-label").value, lr: +$("#cd-lr").value || undefined, ...(sample ? { sample: sample.c, seed: sample.seed } : {}) };
      T.data = await api("/api/code/trace", body);
      cache.clear();
      $("#cd-slider").max = T.data.steps.length - 1;
      $("#cd-meta").innerHTML = `A real training step of the copy in Inside the model (started from ${esc(T.data.source)}), told the mark is “${esc(T.data.label)}”: ${T.data.steps.length} recorded steps, loss ${num(T.data.loss)}. Green is the line that just ran, red the one that runs next.`;
      renderCode(); renderPhases(); go(0);
      DocsShell.refresh();
    } catch (e) { DocsShell.toast(e.message); }
    btn.disabled = false; btn.textContent = "Run a traced training step";
  }
  $("#cd-run").onclick = run;
  let started = false;
  function start() {
    if (started || $('[data-ds-view="code"]').hidden) return;
    started = true;
    fetch("/api/health").then((r) => r.json()).then((h) => { fillSamples(h.classes); run(); });
  }
  document.addEventListener("ds:view", (e) => { if (e.detail === "code") { start(); requestAnimationFrame(arrows); } });
  start();
})();
