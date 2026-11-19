/* InkSolve front end. Plain JS on top of docs-shell. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
  const ago = (t) => { const s = Date.now() / 1000 - t; return s < 60 ? "just now" : s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : new Date(t * 1000).toLocaleDateString(); };
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  let CLASSES = [];

  async function api(path, opts = {}) {
    const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts, body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
    return j;
  }
  const toast = (m) => DocsShell.toast(m);
  function tex(s, display = true) {
    if (!window.katex) return `<code>${esc(s)}</code>`;
    try { return katex.renderToString(s, { displayMode: display, throwOnError: false }); } catch { return `<code>${esc(s)}</code>`; }
  }

  // ================================================================== pen input (shared)
  function Pad(canvas, { onChange, width = 3.2 } = {}) {
    const ctx = canvas.getContext("2d");
    let strokes = [], cur = null, dpr = 1;
    const fit = () => {
      dpr = devicePixelRatio || 1;
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
      draw();
    };
    const pos = (e) => { const r = canvas.getBoundingClientRect(); return [Math.round((e.clientX - r.left) * 10) / 10, Math.round((e.clientY - r.top) * 10) / 10]; };
    function draw() {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineCap = ctx.lineJoin = "round";
      ctx.lineWidth = width;
      ctx.strokeStyle = css("--ink") || "#111";
      for (const s of strokes.concat(cur ? [cur] : [])) {
        ctx.beginPath();
        s.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
        if (s.length === 1) ctx.lineTo(s[0][0] + 0.1, s[0][1] + 0.1);
        ctx.stroke();
      }
    }
    canvas.addEventListener("pointerdown", (e) => { if (e.button > 0) return; e.preventDefault(); canvas.setPointerCapture(e.pointerId); cur = [pos(e)]; draw(); });
    canvas.addEventListener("pointermove", (e) => {
      if (!cur) return;
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      for (const ev of evs) { const p = pos(ev), q = cur[cur.length - 1]; if (Math.hypot(p[0] - q[0], p[1] - q[1]) >= 1.5) cur.push(p); }
      draw();
    });
    const end = () => { if (!cur) return; strokes.push(cur); cur = null; draw(); onChange && onChange(strokes); };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    new ResizeObserver(fit).observe(canvas);
    document.addEventListener("ds:theme", draw);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    return {
      get strokes() { return strokes; },
      set(s) { strokes = s; draw(); onChange && onChange(strokes); },
      undo() { strokes.pop(); draw(); onChange && onChange(strokes); },
      clear() { strokes = []; draw(); onChange && onChange(strokes); },
      draw,
    };
  }

  // ================================================================== SOLVE
  const S = { input: "draw", mode: "solve", last: null, force: {}, photo: null, timer: null, seq: 0 };
  const pad = Pad($("#pad"), {
    onChange(strokes) {
      $("#pad-empty").hidden = strokes.length > 0;
      S.force = {};
      clearTimeout(S.timer);
      if (!strokes.length) { $("#result").innerHTML = ""; boxes($(".pad-wrap"), []); return; }
      S.timer = setTimeout(() => readNow(true), 550);
    },
  });
  $("#undo").onclick = () => pad.undo();
  $("#clear").onclick = () => pad.clear();
  addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "z" && !$('[data-ds-view="solve"]').hidden && S.input === "draw" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); pad.undo(); }
  });
  $("#go-draw").onclick = () => readNow(false);
  $("#go-type").onclick = () => readNow(false);
  $("#typed").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) readNow(false); });
  $("#go-photo").onclick = () => readNow(false);

  $$("#input-nav button").forEach((b) => (b.onclick = () => setInput(b.dataset.input)));
  function setInput(k) {
    S.input = k;
    $$("#input-nav button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.input === k)));
    ["draw", "photo", "type"].forEach((x) => ($(`#in-${x}`).hidden = x !== k));
    $("#result").innerHTML = "";
    if (k === "draw") pad.draw();
    if (k === "type") $("#typed").focus();
  }
  $$("[data-mode]").forEach((b) => (b.onclick = () => {
    S.mode = b.dataset.mode;
    $$("[data-mode]").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    $("#mode-hint").textContent = S.mode === "solve"
      ? "Context may fix a mark: a 1 that must be a 7 for the sum to be true is read as 7."
      : "For marking: the sum is read exactly as written, true or not. Marks the reader is unsure of are outlined for you to check.";
    if (S.last) readNow(true);
  }));

  // photo
  const drop = $("#drop");
  $("#file").onchange = (e) => e.target.files[0] && loadPhoto(e.target.files[0]);
  ["dragover", "dragenter"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadPhoto(f); });
  function loadPhoto(f) {
    const img = new Image();
    img.onload = () => {
      const k = Math.min(1, 1600 / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
      const g = c.getContext("2d"); g.fillStyle = "#fff"; g.fillRect(0, 0, c.width, c.height); g.drawImage(img, 0, 0, c.width, c.height);
      S.photo = c.toDataURL("image/jpeg", 0.92);
      S.photoW = c.width;
      $("#photo").src = S.photo;
      $("#photo-wrap").hidden = false;
      $("#go-photo").disabled = false;
      S.force = {};
      readNow(false);
    };
    img.src = URL.createObjectURL(f);
  }

  async function readNow(preview) {
    const body = { mode: S.mode, preview, force: S.force };
    if (S.input === "draw") { if (!pad.strokes.length) return; body.strokes = pad.strokes; }
    else if (S.input === "photo") { if (!S.photo) return; body.image = S.photo; }
    else { body.text = $("#typed").value; if (!body.text.trim()) return; }
    const seq = ++S.seq;
    try {
      const r = await api("/api/read", { method: "POST", body });
      if (seq !== S.seq) return;
      S.last = r;
      render(r);
      if (!preview) loadRecent();
    } catch (e) {
      if (seq !== S.seq) return;
      $("#result").innerHTML = `<div class="ds-callout">${esc(e.message)}</div>`;
    }
  }

  function boxes(host, glyphs, r, scale = 1) {
    let layer = host.querySelector(".boxes");
    if (!layer) { layer = document.createElement("div"); layer.className = "boxes"; host.appendChild(layer); }
    const changed = new Set((r?.lines || []).flatMap((l) => (l.changed || []).map((c) => c.glyph)));
    const unsure = new Set(S.mode === "grade" ? (r?.lines || []).flatMap((l) => l.unsure || []) : []);
    layer.innerHTML = glyphs.map((g) => {
      const [x0, y0, x1, y1] = g.box.map((v) => v * scale);
      const cls = changed.has(g.i) ? "changed" : unsure.has(g.i) ? "unsure" : "";
      return `<div class="box ${cls}" style="left:${x0 - 3}px;top:${y0 - 3}px;width:${x1 - x0 + 6}px;height:${y1 - y0 + 6}px"><i>${g.i + 1}</i></div>`;
    }).join("");
  }

  function render(r) {
    if (S.input === "draw") boxes($(".pad-wrap"), r.glyphs, r);
    if (S.input === "photo") { const img = $("#photo"); boxes($("#photo-boxes").parentElement, r.glyphs, r, img.clientWidth / S.photoW); }
    const ok = r.lines.length && r.lines.every((l) => l.ok);
    let h = `<h2 id="reading">Reading</h2>`;
    h += `<p class="ds-sub">${r.source === "typed" ? "As typed." : `${r.glyphs.length} marks in ${r.lines.length} line${r.lines.length === 1 ? "" : "s"}, read by ${esc(r.model?.name || "the model")} in ${r.ms} ms.`}</p>`;
    for (const l of r.lines) {
      if (!l.ok) { h += `<div class="ds-callout">${esc(l.error)}</div>`; continue; }
      h += `<div class="reading">${tex(l.latex)}</div>`;
      const bits = [];
      if (r.source !== "typed") bits.push(`Confidence in this reading: <b>${pct(l.confidence, 0)}</b>`);
      if (l.truth) bits.push(`<span class="verdict ${l.truth.true ? "ok" : "no"}">${l.truth.true ? "✓ True as written" : "✗ False as written"}</span>`);
      if (bits.length) h += `<p>${bits.join(" · ")}</p>`;
      for (const w of l.why || []) h += `<div class="ds-callout">${esc(w)} <a href="#/how" onclick="DocsShell.show('how');setTimeout(()=>DocsShell.jump('context'),50);return false">Why?</a></div>`;
      if (S.mode === "grade" && l.unsure?.length) h += `<div class="ds-callout">Check mark${l.unsure.length > 1 ? "s" : ""} ${l.unsure.map((i) => i + 1).join(", ")}: the reader is less than 80% sure of ${l.unsure.length > 1 ? "their" : "its"} shape, and in this mode the sum doesn't help decide.</div>`;
      if (l.alternatives?.length) h += `<p class="alts">Also possible: ${l.alternatives.slice(0, 3).map((a) => `<span>${esc(a.text)}</span> (${pct(a.p, 0)}${a.truth === false ? ", false" : ""})`).join(", ")}</p>`;
    }
    if (r.glyphs.length) {
      h += `<h2 id="marks">Marks</h2><p class="ds-sub">What each mark looks like on its own. If one is wrong, choose what it really is: InkSolve re-reads the line and keeps the mark as a sample of your handwriting.</p><div class="marks">`;
      for (const g of r.glyphs) {
        const changed = r.lines.some((l) => (l.changed || []).some((c) => c.glyph === g.i));
        const unsure = S.mode === "grade" && r.lines.some((l) => (l.unsure || []).includes(g.i));
        h += `<div class="mark ${changed ? "changed" : unsure ? "unsure" : ""}"><div class="top"><img src="${g.png}" alt="mark ${g.i + 1}"><span class="rd">${esc(g.read ?? "?")}</span><span class="n">${g.i + 1}</span></div>`;
        if (g.fixed) h += `<div class="ds-hint" style="margin:6px 0 0">${g.fixed === "=" ? "two bars: =" : "one bar: −"}</div>`;
        else {
          g.top.forEach((t, k) => { h += `<div class="bar ${k === 0 ? "first" : ""}"><b>${esc(t.c)}</b><span><i style="width:${Math.max(1, t.p * 100)}%"></i></span><small>${Math.round(t.p * 100)}%</small></div>`; });
          h += `<select data-glyph="${g.i}" aria-label="Correct mark ${g.i + 1}"><option value="">${g.forced ? "Corrected ✓" : "Correct it…"}</option>${CLASSES.map((c) => `<option value="${esc(c)}">${esc(c === "x" ? "x or ×" : c)}</option>`).join("")}</select>`;
        }
        h += `</div>`;
      }
      h += `</div>`;
    }
    if (ok && r.solution) h += solutionHtml(r.solution, r.lines);
    $("#result").innerHTML = h;
    if (window.Guide && r.lines.length) {
      const l = r.lines[0];
      Guide.context("solve", { kind: "reading", text: r.lines.map((x) => x.text || "?").join(" ; "), marks: r.glyphs.length, conf: pct(l.confidence, 0), truth: l.truth ? l.truth.true : null,
        mode: S.mode, changed: r.lines.flatMap((x) => x.why || []) });
    }
    $$("#result select[data-glyph]").forEach((sel) => (sel.onchange = async () => {
      const gi = +sel.dataset.glyph, label = sel.value;
      if (!label) return;
      S.force[gi] = label;
      if (r.source !== "typed") {
        try { await api("/api/correct", { method: "POST", body: { read_id: r.id, glyph: gi, label } }); toast(`Saved mark ${gi + 1} as “${label}” in your handwriting samples.`); } catch (e) { toast(e.message); }
      }
      readNow(true);
    }));
    DocsShell.refresh();
  }

  function solutionHtml(sols, lines) {
    let h = "";
    sols.forEach((s, k) => {
      const title = sols.length > 1 ? ` (line ${k + 1})` : "";
      if (s.kind === "error") { h += `<div class="ds-callout">${esc(s.note)}</div>`; return; }
      const kindName = { arithmetic: "Working", statement: "Checking the statement", linear: "Working", quadratic: "Working", equation: "Working", rearrange: "Rearranging", system: "Solving them together", expression: "Rearranging", identity: "Working", contradiction: "Working" }[s.kind] || "Working";
      if (s.steps?.length) {
        h += `<h2 id="working-${k}">${kindName}${title}</h2><ol class="steps">`;
        for (const st of s.steps) h += `<li>${st.label ? `<span class="lab">${esc(st.label)}</span>` : ""}<span class="math">${tex(st.latex, false)}</span></li>`;
        h += `</ol>`;
      }
      if (s.answer) {
        h += `<h2 id="answer-${k}">Answer${title}</h2><div class="answer">${tex(s.answer.latex)}</div>`;
        if (s.answer.decimal) h += `<p class="ds-meta-line">≈ ${esc(s.answer.decimal)}</p>`;
      }
      if (s.note) h += `<p>${s.note.includes("\\") ? tex(s.note, false) : esc(s.note)}</p>`;
      if (s.checks?.length) {
        h += `<h3 id="check-${k}">Check</h3><table><thead><tr><th>With</th><th>Left side</th><th>Right side</th><th></th></tr></thead><tbody>`;
        for (const c of s.checks) h += `<tr><td>${tex(c.with, false)}</td><td>${tex(c.lhs, false)}</td><td>${tex(c.rhs, false)}</td><td class="${c.ok ? "good" : ""}">${c.ok ? "✓ equal" : "✗ not equal"}</td></tr>`;
        h += `</tbody></table>`;
      }
    });
    return h;
  }

  // examples and recent
  const EX = [["ambiguous-7", "6 + 1 = 7, with a 7 like a 1"], ["linear", "2x + 3 = 11"], ["quadratic", "x² − 5x + 6 = 0"], ["times", "3x4: letter or times?"], ["decimal", "3.5 + 1 = 4.5"], ["system", "Two equations"]];
  $("#examples").innerHTML = EX.map(([k, t]) => `<li><button data-ex="${k}">${esc(t)}</button></li>`).join("");
  $$("#examples button").forEach((b) => (b.onclick = async () => {
    setInput("draw");
    const ex = await api(`/api/example/${b.dataset.ex}`);
    const r = $("#pad").getBoundingClientRect();
    const xs = ex.strokes.flat(), maxX = Math.max(...xs.map((p) => p[0])), maxY = Math.max(...xs.map((p) => p[1]));
    const k = Math.min(1.5, (r.width - 30) / maxX, (r.height - 20) / maxY);
    pad.set(ex.strokes.map((s) => s.map(([x, y]) => [x * k, y * k])));
    clearTimeout(S.timer);
    readNow(false);
  }));
  async function loadRecent() {
    try {
      const seen = new Set();
      const hs = (await api("/api/history?limit=40")).filter((h) => h.text && !seen.has(h.text) && seen.add(h.text)).slice(0, 6);
      $("#recent").innerHTML = hs.length ? hs.map((h) => `<li><button data-text="${esc(h.text)}" title="${esc(h.source)} · ${ago(h.time)}">${esc(h.text)}</button></li>`).join("") : `<li class="ds-hint">Nothing read yet.</li>`;
      $$("#recent button").forEach((b) => (b.onclick = () => { setInput("type"); $("#typed").value = b.dataset.text.split(" ; ").join("\n"); readNow(true); }));
    } catch { /* offline */ }
  }

  // ================================================================== YOUR HANDWRITING
  const H = { target: "7", status: null };
  const cpad = Pad($("#cpad"), { width: 4 });
  const LOOK = ["1", "7", "b", "6", "z", "2", "x", "+", "5", "3", "4", "9", "(", "/"];
  function nextTarget() {
    const counts = H.status?.counts || {};
    const mine = (c) => counts[c]?.mine || 0;
    const pool = CLASSES.slice().sort((a, b) => (mine(a) - (LOOK.includes(a) ? 6 : 0)) - (mine(b) - (LOOK.includes(b) ? 6 : 0)) || Math.random() - 0.5);
    setTarget(pool[Math.floor(Math.random() * Math.min(4, pool.length))]);
  }
  function setTarget(c) {
    H.target = c;
    $("#target").textContent = c === "x" ? "x (or ×)" : c;
    $$("#keys button").forEach((b) => b.classList.toggle("on", b.dataset.c === c));
    cpad.clear();
  }
  async function saveSample() {
    if (!cpad.strokes.length) { toast("Write the symbol first."); return; }
    try {
      await api("/api/samples", { method: "POST", body: { label: H.target, strokes: cpad.strokes } });
      $("#csaved").textContent = `Saved a “${H.target}”.`;
      await loadHand();
      nextTarget();
    } catch (e) { toast(e.message); }
  }
  $("#csave").onclick = saveSample;
  $("#cclear").onclick = () => cpad.clear();
  $("#cskip").onclick = nextTarget;
  addEventListener("keydown", (e) => {
    if ($('[data-ds-view="hand"]').hidden || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    if (e.key === "Enter") { e.preventDefault(); saveSample(); }
    if (e.key === "Escape") cpad.clear();
  });

  async function loadHand() {
    const st = await api("/api/datasets");
    H.status = st;
    const src = st.sources;
    const mineTotal = Object.values(st.counts).reduce((a, c) => a + (c.mine || 0), 0);
    $("#hand-meta").textContent = `${mineTotal} samples of your handwriting · MNIST ${src.mnist.present ? "downloaded" : "not downloaded"} · HASYv2 ${src.hasy.present ? "downloaded" : "not downloaded"}`;
    $("#keys").innerHTML = CLASSES.map((c) => `<button data-c="${esc(c)}" class="${c === H.target ? "on" : ""}">${esc(c)}<small>${st.counts[c]?.mine || ""}</small></button>`).join("");
    $$("#keys button").forEach((b) => (b.onclick = () => setTarget(b.dataset.c)));
    let t = `<div class="tablewrap"><table><thead><tr><th>Symbol</th><th class="num">Synthetic</th><th class="num">MNIST</th><th class="num">HASYv2</th><th class="num">Yours</th></tr></thead><tbody>`;
    for (const c of CLASSES) {
      const k = st.counts[c] || {};
      t += `<tr><td><b>${esc(c)}</b></td><td class="num">any number</td><td class="num">${k.mnist ? k.mnist.toLocaleString() : "–"}</td><td class="num">${k.hasy ? k.hasy.toLocaleString() : "–"}</td><td class="num">${k.mine || "–"}</td></tr>`;
    }
    $("#counts").innerHTML = t + "</tbody></table></div>";
    const sm = await api("/api/samples?limit=120");
    $("#gallery").innerHTML = sm.length
      ? `<div class="gallery ds-no-md">${sm.map((s) => `<figure><img src="${s.png}" alt="${esc(s.label)}"><figcaption>${esc(s.label)}${s.source === "correction" ? " ✎" : ""}</figcaption><button data-del="${s.id}" title="Delete" aria-label="Delete">×</button></figure>`).join("")}</div><p class="ds-hint">✎ marks a correction made on the Solve tab.</p>`
      : emptyState("No samples yet", "Write a few symbols above. Start with the ones your hand makes look alike.");
    $$("#gallery [data-del]").forEach((b) => (b.onclick = async () => { await api(`/api/samples/${b.dataset.del}`, { method: "DELETE" }); loadHand(); }));
    const ds = [["mnist", "MNIST", "70,000 handwritten digits from about 500 writers. About 11 MB.", src.mnist],
      ["hasy", "HASYv2", "168,000 handwritten symbols from many writers; InkSolve keeps the operators, brackets, letters and digits. About 32 MB.", src.hasy]];
    $("#datasets-body").innerHTML = `<table><thead><tr><th>Dataset</th><th>What</th><th class="num">Training</th><th class="num">Test</th><th></th></tr></thead><tbody>${ds.map(([k, n, d, s]) =>
      `<tr><td><b>${n}</b></td><td>${d}</td><td class="num">${s.present ? s.train.toLocaleString() : "–"}</td><td class="num">${s.present ? s.test.toLocaleString() : "–"}</td><td>${s.present ? '<span class="pill">Ready</span>' : `<button class="ds-btn" data-dl="${k}">Download</button>`}</td></tr>`).join("")}</tbody></table>`;
    $$("[data-dl]").forEach((b) => (b.onclick = async () => { try { await api(`/api/datasets/${b.dataset.dl}/download`, { method: "POST" }); pollJobs(); toast("Downloading in the background."); } catch (e) { toast(e.message); } }));
    DocsShell.refresh();
  }

  function emptyState(title, line, btn = "") {
    return `<div class="ds-empty"><svg viewBox="0 0 100 100" aria-hidden="true"><g stroke-width="3" fill="none"><path d="M22 22h56" stroke="#e4572e"/><path d="M22 22l56 56" stroke="#e4572e"/><path d="M78 22v56" stroke="#2e9e6b"/><path d="M22 78h56" stroke="#2f6fde"/><path d="M22 22v56" stroke="#e0a03a"/></g><g fill="#f6d2a8"><circle cx="22" cy="22" r="7"/><circle cx="78" cy="22" r="7"/><circle cx="22" cy="78" r="7"/><circle cx="78" cy="78" r="7"/></g></svg><b>${esc(title)}</b>${esc(line)}${btn ? `<p>${btn}</p>` : ""}</div>`;
  }

  // ================================================================== TRAIN
  const T = { selected: null };
  async function loadModels() {
    const { active, models } = await api("/api/models");
    $("#train-meta").textContent = `${models.length} model${models.length === 1 ? "" : "s"} · in use: ${models.find((m) => m.id === active)?.name || "none"}`;
    $("#t-init").innerHTML = `<option value="">Nothing (from scratch)</option>` + models.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("");
    if (!models.length) { $("#models-body").innerHTML = emptyState("No models yet", "Train one from the panel on the left."); $("#model-detail-body").innerHTML = ""; return; }
    const ev = (m, k) => m.eval?.[k]?.acc;
    let h = `<div class="tablewrap"><table><thead><tr><th>Model</th><th class="num">MNIST</th><th class="num">HASYv2</th><th class="num">Yours</th><th class="num">Synthetic</th><th class="num">1→7</th><th class="num">ECE</th><th></th></tr></thead><tbody>`;
    for (const m of models) {
      const p = m.eval?.mnist?.pairs?.["1|7"] || m.eval?.synthetic?.pairs?.["1|7"];
      h += `<tr class="${m.id === T.selected ? "sel" : ""}"><td><button class="link" data-m="${m.id}">${esc(m.name)}</button><br><small class="ds-hint">${ago(m.created)} · ${m.n_train?.toLocaleString() || "?"} examples · ${m.steps} steps</small></td>
        <td class="num">${pct(ev(m, "mnist"))}</td><td class="num">${pct(ev(m, "hasy"))}</td><td class="num">${pct(ev(m, "mine"))}</td><td class="num">${pct(ev(m, "synthetic"))}</td>
        <td class="num">${pct(p?.a_as_b, 2)}</td><td class="num">${m.eval?.mnist ? m.eval.mnist.ece.toFixed(3) : m.eval?.synthetic ? m.eval.synthetic.ece.toFixed(3) : "–"}</td>
        <td>${m.id === active ? '<span class="pill">In use</span>' : `<button class="ds-btn" data-use="${m.id}">Use</button>`}</td></tr>`;
    }
    $("#models-body").innerHTML = h + "</tbody></table></div><p class=\"ds-hint\">1→7: share of real 1s read as 7 on shape alone. ECE: how far its confidence is from its real accuracy (0 is perfectly honest).</p>";
    $$("[data-use]").forEach((b) => (b.onclick = async () => { await api(`/api/models/${b.dataset.use}/activate`, { method: "POST" }); toast("Now reading with this model."); loadModels(); health(); }));
    $$("[data-m]").forEach((b) => (b.onclick = () => { T.selected = b.dataset.m; loadModels(); }));
    showModel(T.selected || active || models[0].id, active);
  }
  async function showModel(id, active) {
    const m = await api(`/api/models/${id}`);
    let h = `<p><b>${esc(m.name)}</b>${m.id === active ? ' <span class="pill">In use</span>' : ""} · temperature ${m.temperature?.toFixed(2)} · ${Math.round(m.seconds)} s to train${m.init_from ? ` · started from ${esc(m.init_from)}` : ""}${m.augment ? " · augmented" : ""}</p>`;
    h += `<p class="ds-meta-line">Examples per symbol: ${Object.entries(m.counts || {}).map(([k, v]) => `${k} ${Math.max(...Object.values(v))}`).join(", ")}</p>`;
    if (m.curve?.length) {
      const pts = m.curve.filter((r) => r.val_acc != null);
      h += `<h3 id="training-curve">Training</h3><div id="curve"></div>`;
      setTimeout(() => {
        const el = $("#curve");
        if (el && m.curve.length) lineChart(el, { title: "Loss while training", x: m.curve.map((r) => r.step), xName: "step", xLabel: "training step", yFmt: (v) => v.toFixed(2), series: [{ name: "loss", color: css("--s1"), values: m.curve.map((r) => r.loss) }] });
      });
      if (pts.length) h += `<p class="ds-meta-line">Held-out accuracy during training: ${pts.map((r) => pct(r.val_acc)).join(" → ")}</p>`;
    }
    const sets = Object.entries(m.eval || {}).filter(([, v]) => v);
    if (sets.length) {
      h += `<h3 id="per-symbol">Per symbol</h3><div class="tablewrap"><table><thead><tr><th>Symbol</th>${sets.map(([k]) => `<th class="num">${esc(k)}</th>`).join("")}</tr></thead><tbody>`;
      for (const c of CLASSES) h += `<tr><td><b>${esc(c)}</b></td>${sets.map(([, v]) => `<td class="num">${v.per_class?.[c] ? pct(v.per_class[c].acc) : "–"}</td>`).join("")}</tr>`;
      h += `</tbody></table></div><h3 id="look-alikes">Look-alikes</h3><div class="tablewrap"><table><thead><tr><th>Pair</th>${sets.map(([k]) => `<th class="num">${esc(k)}</th>`).join("")}</tr></thead><tbody>`;
      const pairs = [...new Set(sets.flatMap(([, v]) => Object.keys(v.pairs || {})))];
      for (const p of pairs) {
        const [a, b] = p.split("|");
        h += `<tr><td>${esc(a)} read as ${esc(b)} / ${esc(b)} read as ${esc(a)}</td>${sets.map(([, v]) => { const q = v.pairs?.[p]; return `<td class="num">${q ? `${pct(q.a_as_b, 1)} / ${pct(q.b_as_a, 1)}` : "–"}</td>`; }).join("")}</tr>`;
      }
      h += `</tbody></table></div>`;
    }
    h += `<div class="row">${m.id !== active ? `<button class="ds-btn" id="m-del">Delete this model</button>` : ""}</div>`;
    $("#model-detail-body").innerHTML = h;
    const del = $("#m-del");
    if (del) del.onclick = async () => { if (!confirm(`Delete “${m.name}”?`)) return; try { await api(`/api/models/${m.id}`, { method: "DELETE" }); T.selected = null; loadModels(); } catch (e) { toast(e.message); } };
    DocsShell.refresh();
  }
  $("#t-go").onclick = async () => {
    const body = { synthetic: +$("#t-syn").value || 0, mnist: +$("#t-mnist").value || 0, hasy: +$("#t-hasy").value || 0, mine: $("#t-mine").value || null, mine_weight: 5,
      steps: +$("#t-steps").value || 2000, augment: $("#t-aug").checked, init_from: $("#t-init").value || null, name: $("#t-name").value };
    try { await api("/api/train", { method: "POST", body }); toast("Training started."); pollJobs(); } catch (e) { toast(e.message); }
  };

  // ================================================================== EXPERIMENT
  const X = { preset: "quick", runs: [], current: null };
  const PRESET_HINT = { quick: "10, 50 and 250 examples per symbol, 500 steps each: about 10 minutes on a Mac.", full: "5 to 1,000 examples per symbol, 900 steps each: about 45 minutes on a Mac." };
  $$("[data-preset]").forEach((b) => (b.onclick = () => { X.preset = b.dataset.preset; $$("[data-preset]").forEach((x) => x.setAttribute("aria-checked", String(x === b))); $("#preset-hint").textContent = PRESET_HINT[X.preset]; }));
  $("#preset-hint").textContent = PRESET_HINT.quick;
  $("#e-go").onclick = async () => {
    const recipes = $$("[data-recipe]").filter((c) => c.checked).map((c) => c.dataset.recipe);
    if (!recipes.length) return toast("Choose at least one recipe.");
    try { await api("/api/experiment", { method: "POST", body: { preset: X.preset, recipes } }); toast("Experiment started; it runs in the background."); pollJobs(); } catch (e) { toast(e.message); }
  };
  const RNAME = { scratch: "From scratch", augment: "+ augmentation", pretrain: "+ synthetic pretraining" };
  const RCOL = () => ({ scratch: css("--s1"), augment: css("--s2"), pretrain: css("--s3") });
  async function loadExperiments(live) {
    X.runs = await api("/api/experiments");
    $("#exp-nav").innerHTML = X.runs.map((r, i) => `<li><button data-run="${i}" aria-selected="${(X.current ?? 0) === i}">${new Date(r.created * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} · ${r.rows.length} runs</button></li>`).join("");
    $$("[data-run]").forEach((b) => (b.onclick = () => { X.current = +b.dataset.run; loadExperiments(); }));
    const run = live ? { rows: live.rows, config: {}, examples: [], live: true } : X.runs[X.current ?? 0];
    const body = $("#exp-body");
    let h = `<h1 class="ds-no-toc">How few examples?</h1>`;
    if (!run || !run.rows?.length) {
      body.innerHTML = h + `<p class="ds-meta-line">Not run yet</p><p>This trains the symbol reader again and again with fewer and fewer real examples of each symbol, with and without the tricks for needing fewer, and measures each one on writers it has never seen, both one mark at a time and on whole sums, with and without context.</p>` + emptyState("No results yet", "Run the experiment from the panel on the left.");
      DocsShell.refresh();
      return;
    }
    const rows = run.rows, ns = [...new Set(rows.map((r) => r.n))].sort((a, b) => a - b), recipes = [...new Set(rows.map((r) => r.recipe))];
    const at = (n, rc, k) => rows.find((r) => r.n === n && r.recipe === rc)?.[k] ?? null;
    h += `<p class="ds-meta-line">${run.live ? "Running now · " : `${new Date(run.created * 1000).toLocaleString()} · ${Math.round(run.seconds / 60)} min · `}${rows.length} trained readers${run.n_test_symbols ? ` · ${run.n_test_symbols.toLocaleString()} test marks · ${run.n_expr} test sums` : ""}${run.config?.steps ? ` · ${run.config.steps} steps each` : ""}</p>`;
    // headline finding
    const best = (rc) => rows.filter((r) => r.recipe === rc);
    const small = ns[0], big = ns[ns.length - 1];
    if (recipes.includes("scratch") && recipes.length > 1) {
      const last = recipes[recipes.length - 1];
      const target = at(big, "scratch", "symbol_acc");
      const match = ns.find((n) => (at(n, last, "symbol_acc") ?? 0) >= (target ?? 2) - 0.005);
      h += `<p>With ${small} examples per symbol, a reader trained from scratch gets <b>${pct(at(small, "scratch", "symbol_acc"))}</b> of unseen marks right; ${RNAME[last].replace("+ ", "adding ").toLowerCase()} makes that <b>${pct(at(small, last, "symbol_acc"))}</b>.${match && match < big ? ` It comes within half a point of what from scratch reaches with ${big} examples using only <b>${match}</b>: ${Math.round(big / match)} times fewer.` : ""}
        On whole sums, context lifts the from-scratch reader at ${small} per symbol from <b>${pct(at(small, "scratch", "shape_acc"), 0)}</b> to <b>${pct(at(small, "scratch", "context_acc"), 0)}</b> read exactly right.</p>`;
    }
    h += `<h2 id="symbols">Single marks</h2><p class="ds-sub">Share of unseen marks read right, by examples per symbol (log scale).</p><div id="c-sym"></div>`;
    h += `<h2 id="sums">Whole sums, with and without context</h2><p class="ds-sub">Share of true sums read exactly right. Solid: with context. Dashed: on shape alone, where every mark has to be right by itself.</p><div id="c-sum"></div>`;
    const hidS = rows.map((r) => r.hidden_solve), hidG = rows.map((r) => r.hidden_grade);
    h += `<h2 id="marking">Marking work</h2><p class="ds-sub">False sums that came out as true ones, so a mistake would have been hidden.</p>
      <table><thead><tr><th>Mode</th><th class="num">Mistakes hidden (average)</th><th class="num">Worst run</th></tr></thead><tbody>
      <tr><td>Solve (uses whether the sum is true)</td><td class="num">${pct(avg(hidS))}</td><td class="num">${pct(Math.max(...hidS))}</td></tr>
      <tr><td>Mark work (doesn't)</td><td class="num">${pct(avg(hidG))}</td><td class="num">${pct(Math.max(...hidG))}</td></tr></tbody></table>
      <div class="ds-callout">This is why marking mode never uses truth. Any “hidden” count above zero there comes from the reader misreading a mark on shape alone, not from context.</div>`;
    if (run.examples?.length) {
      h += `<h2 id="fixed">Sums context fixed</h2><p class="ds-sub">Real held-out marks the reader got wrong on shape alone and right with context. The outlined mark is the one context changed.</p><div class="examples-fixed ds-no-md">`;
      for (const e of run.examples) h += `<div class="fixed-ex">${e.images.map((src, j) => `<img src="${src}" class="${e.shape[j] !== e.truth[j] ? "wrong" : ""}" alt="${esc(e.truth[j])}">`).join("")}<span class="txt">shape alone: <code>${esc(e.shape)}</code> → with context: <code>${esc(e.read)}</code> · ${e.n} per symbol</span></div>`;
      h += `</div>`;
    }
    h += `<h2 id="all-runs">All runs</h2><div class="tablewrap"><table><thead><tr><th class="num">Per symbol</th><th>Recipe</th><th class="num">Marks</th><th class="num">Digits</th><th class="num">1→7</th><th class="num">7→1</th><th class="num">ECE</th><th class="num">Sums, shape</th><th class="num">Sums, context</th><th class="num">Grading reads</th></tr></thead><tbody>`;
    for (const r of rows) h += `<tr><td class="num">${r.n}</td><td style="white-space:nowrap">${RNAME[r.recipe]}</td><td class="num">${pct(r.symbol_acc)}</td><td class="num">${pct(r.digits_acc)}</td><td class="num">${pct(r.one_as_seven)}</td><td class="num">${pct(r.seven_as_one)}</td><td class="num">${r.ece.toFixed(3)}</td><td class="num">${pct(r.shape_acc, 0)}</td><td class="num">${pct(r.context_acc, 0)}</td><td class="num">${pct(r.grade_acc, 0)}</td></tr>`;
    h += `</tbody></table></div>`;
    if (run.sources) {
      const by = {}; Object.entries(run.sources).forEach(([c, s]) => (by[s] = (by[s] || []).concat(c)));
      h += `<p class="ds-hint">Real examples from: ${Object.entries(by).map(([s, cs]) => `${s} (${cs.join(" ")})`).join("; ")}.</p>`;
    }
    body.innerHTML = h;
    const col = RCOL();
    if (ns.length > 1) {
      lineChart($("#c-sym"), { title: "Symbol accuracy", x: ns, logX: true, xName: "examples per symbol:", xLabel: "real examples per symbol", yFmt: (v) => pct(v, 0),
        series: recipes.map((rc) => ({ name: RNAME[rc], color: col[rc], values: ns.map((n) => at(n, rc, "symbol_acc")) })) });
      const sr = recipes.slice(-2);
      lineChart($("#c-sum"), { title: "Sums read right", x: ns, logX: true, xName: "examples per symbol:", xLabel: "real examples per symbol", yFmt: (v) => pct(v, 0),
        series: sr.flatMap((rc) => [{ name: `${RNAME[rc]}, context`, color: col[rc], values: ns.map((n) => at(n, rc, "context_acc")) },
          { name: `${RNAME[rc]}, shape only`, color: col[rc], dash: true, values: ns.map((n) => at(n, rc, "shape_acc")) }]) });
    } else { $("#c-sym").innerHTML = `<p class="ds-hint">Charts appear once two sizes have run.</p>`; $("#c-sum").innerHTML = ""; }
    DocsShell.refresh();
  }
  const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);

  // ================================================================== jobs banner
  let polling = null;
  async function pollJobs() {
    clearTimeout(polling);
    let jobs = [];
    try { jobs = await api("/api/jobs"); } catch { polling = setTimeout(pollJobs, 5000); return; }
    const run = jobs.find((j) => j.status === "running") || jobs.find((j) => j.status === "queued");
    const b = $("#banner");
    if (run) {
      const full = await api(`/api/jobs/${run.id}`).catch(() => run);
      b.hidden = false;
      b.innerHTML = `<b>${esc(full.title)}</b><span class="progress"><i style="width:${Math.round(full.progress * 100)}%"></i></span><span class="msg">${esc(full.message)}</span><button id="stop">Stop</button>`;
      $("#stop").onclick = () => api(`/api/jobs/${run.id}/stop`, { method: "POST" }).then(pollJobs);
      if (full.kind === "train" && full.live?.curve && !$('[data-ds-view="train"]').hidden) {
        $("#train-live").innerHTML = `<h2 id="training-now">Training now</h2><div id="live-curve"></div>`;
        lineChart($("#live-curve"), { title: "loss", x: full.live.curve.map((r) => r.step), xName: "step", xLabel: "training step", yFmt: (v) => v.toFixed(2), series: [{ name: "loss", color: css("--s1"), values: full.live.curve.map((r) => r.loss) }] });
      }
      if (full.kind === "experiment" && full.live?.rows?.length && !$('[data-ds-view="experiment"]').hidden && X.current == null) loadExperiments(full.live);
      S.jobWas = run.id;
      polling = setTimeout(pollJobs, 1500);
    } else {
      b.hidden = true;
      if (S.jobWas) {
        const j = jobs.find((x) => x.id === S.jobWas);
        S.jobWas = null;
        $("#train-live").innerHTML = "";
        if (j) toast(j.status === "done" ? `Finished: ${j.title}` : j.status === "error" ? `Failed: ${j.error}` : `Stopped: ${j.title}`);
        refreshView();
        health();
      }
      polling = setTimeout(pollJobs, 8000);
    }
  }

  // ================================================================== wiring
  async function health() {
    try {
      const h = await api("/api/health");
      CLASSES = h.classes;
      $("#model-line").innerHTML = h.model ? `Reading with <b>${esc(h.model.name)}</b> (${esc(h.device)}).` : "No model yet: train one on the Train tab.";
    } catch { $("#model-line").textContent = "InkSolve's server isn't answering."; }
  }
  function refreshView() {
    const v = location.hash.replace("#/", "") || "solve";
    if (v === "hand") loadHand().then(() => { if (!H.started) { H.started = true; nextTarget(); } });
    if (v === "train") loadModels();
    if (v === "experiment") loadExperiments();
  }
  document.addEventListener("ds:view", refreshView);

  DocsShell.init({
    key: "inksolve",
    search: {
      placeholder: "Search",
      sources: [async (q) => (await api(`/api/search?q=${encodeURIComponent(q)}`)).map((r) => ({
        group: r.group, title: r.title, sub: r.sub, icon: r.kind === "history" ? "∑" : r.kind === "model" ? "◆" : "▲",
        run: () => {
          if (r.kind === "history") { DocsShell.show("solve"); setInput("type"); $("#typed").value = r.title.split(" ; ").join("\n"); readNow(true); }
          else if (r.kind === "model") { T.selected = r.id; DocsShell.show("train"); }
          else { X.current = X.runs.findIndex((x) => x.id === r.id); DocsShell.show("experiment"); }
        },
      }))],
    },
  });
  health().then(() => { refreshView(); loadRecent(); });
  pollJobs();
})();
