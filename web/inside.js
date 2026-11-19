/* Inside the model: every weight of InkSolve's symbol reader as a block in 3D, with the
   forward pass, the backward pass and the update played over it. Data comes from
   /api/inside/* (server/inside.py); the network is the real one. */
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/three/addons/OrbitControls.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtInt = (n) => Math.round(n).toLocaleString();
const sci = (v, d = 3) => (v == null ? "–" : v === 0 ? "0" : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? String(+v.toPrecision(Math.max(2, d))) : v.toExponential(Math.max(1, Math.min(d, 3))));
const sgn = (v, d = 5) => (v >= 0 ? "+" : "−") + sci(Math.abs(v), d);
const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);

async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
  return j;
}

// ------------------------------------------------------------------ decoding
const HALF = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
})();
function bytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }
function f16(b64) { if (!b64) return null; const u = bytes(b64), h = new Uint16Array(u.buffer), o = new Float32Array(h.length); for (let i = 0; i < h.length; i++) o[i] = HALF[h[i]]; return o; }
function u8(b64, max = 1) { const u = bytes(b64), o = new Float32Array(u.length); for (let i = 0; i < u.length; i++) o[i] = (u[i] / 255) * max; return o; }
const maxAbs = (a) => { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m || 1; };

// ------------------------------------------------------------------ state
const S = {
  data: null, layers: {}, acts: {}, stages: [], mode: "value", exag: 300, built: false, busy: false, playing: false,
  strokes: [], sample: null, labelTouched: false, pinned: null, history: [], phase: null,
};

// ------------------------------------------------------------------ three.js scene
let renderer, scene, camera, controls, box, marker, pulse, raycaster, floorPlane;
const tweens = [];
function tween(dur, fn, done) { tweens.push({ t0: performance.now(), dur, fn, done }); }
const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);

function palette() {
  const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  return {
    dark,
    bg: new THREE.Color(css("--scene-bg") || (dark ? "#101012" : "#f4f3ef")),
    neg: new THREE.Color(dark ? "#3987e5" : "#2a78d6"),
    pos: new THREE.Color(dark ? "#e8743f" : "#eb6834"),
    mid: new THREE.Color(dark ? "#3a3a3e" : "#d4d2cc"),
    hot: new THREE.Color(dark ? "#f2c463" : "#c98500"),
    hot2: new THREE.Color(dark ? "#fff3c4" : "#7a4b00"),
    floor: new THREE.Color(dark ? "#1b1b1e" : "#e9e7e1"),
    floorW: new THREE.Color(dark ? "#211d16" : "#efe6d4"),
  };
}
let P;

function init3d() {
  const canvas = $("#in-canvas");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  scene = new THREE.Scene();
  P = palette();
  scene.background = P.bg;
  camera = new THREE.PerspectiveCamera(42, 1, 0.5, 6000);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.addEventListener("change", redraw);
  controls.maxPolarAngle = Math.PI * 0.495;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6b6b6b, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.3);
  sun.position.set(-0.4, 1, 0.7);
  scene.add(sun);
  box = new THREE.BoxGeometry(1, 1, 1);
  box.translate(0, 0.5, 0);                 // bars grow up from the floor
  const idx = Array.from(box.getIndex().array);
  box.setIndex([...idx.slice(0, 18), ...idx.slice(24)]);   // drop the face that sits on the floor: a sixth fewer triangles
  marker = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true }));
  marker.geometry.translate(0, 0.5, 0);
  marker.visible = false;
  scene.add(marker);
  pulse = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xe0a03a, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
  pulse.rotation.y = Math.PI / 2;
  pulse.visible = false;
  scene.add(pulse);
  raycaster = new THREE.Raycaster();
  floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  new ResizeObserver(resize).observe(canvas);
  resize();
  canvas.addEventListener("pointermove", hover);
  canvas.addEventListener("pointerleave", () => { $("#in-tip").hidden = true; marker.visible = !!S.pinned; dirty = true; });
  let down = null;
  canvas.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  canvas.addEventListener("pointerup", (e) => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) pick(e); down = null; });
  document.addEventListener("ds:theme", retheme);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
  requestAnimationFrame(loop);
}
function retheme() { if (!scene) return; P = palette(); scene.background = P.bg; dirty = true; S.stages.forEach((s) => { if (s.floor) s.floor.material.color = s.kind === "w" ? P.floorW : P.floor; }); recolorAll(); }
function resize() {
  const c = $("#in-canvas"), r = c.getBoundingClientRect();
  if (!r.width) return;
  renderer.setSize(r.width, r.height, false);
  dirty = true;
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
function visible() { const v = $('[data-ds-view="inside"]'); return v && !v.hidden; }

let dirty = true;                       // draw only when something changed: the scene is large
const redraw = () => (dirty = true);
function loop(now) {
  requestAnimationFrame(loop);
  if (!visible()) return;
  const moving = controls.update();
  if (!dirty && !tweens.length && !moving) return;
  dirty = false;
  for (let i = tweens.length - 1; i >= 0; i--) {
    const t = tweens[i], p = Math.min(1, (now - t.t0) / t.dur);
    t.fn(p);
    if (p >= 1) { tweens.splice(i, 1); t.done && t.done(); }
  }
  renderer.render(scene, camera);
  placeLabels();
}

// ------------------------------------------------------------------ layout: one stage per weight tensor / activation
const GAP = 14;
function buildStages(d) {
  S.stages.forEach((s) => { scene.remove(s.mesh); s.floor && scene.remove(s.floor); });
  S.stages = [];
  const L = Object.fromEntries(d.layers.map((l) => [l.name, l]));
  const A = Object.fromEntries(d.acts.map((a) => [a.name, a]));
  const plan = [["input", "in"], ["conv1", "w"], ["conv1", "a"], ["conv2", "w"], ["conv2", "a"], ["pool1", "a"], ["conv3", "w"], ["conv3", "a"], ["conv4", "w"], ["conv4", "a"],
    ["pool2", "a"], ["conv5", "w"], ["conv5", "a"], ["pool3", "a"], ["fc1", "w"], ["fc1", "a"], ["fc2", "w"], ["fc2", "out"]];
  let x = 0;
  for (const [name, kind] of plan) {
    const st = { name, kind };
    if (kind === "in") { st.shape = [1, 28, 28]; gridMaps(st, 1, 28, 28, 1.4); }
    else if (kind === "w") {
      const l = L[name]; st.layer = l; st.shape = l.shape; st.n = l.shape.reduce((a, b) => a * b, 1);
      if (l.kind === "conv") {
        const [O, I] = l.shape; st.cols = I * 4 - 1; st.rows = O * 4 - 1;
        st.cs = Math.min(1, 80 / Math.max(st.cols, st.rows));
        st.cell = (i) => { const kx = i % 3, ky = ((i / 3) | 0) % 3, ii = ((i / 9) | 0) % I, o = (i / (9 * I)) | 0; return [ii * 4 + kx, o * 4 + ky]; };
        st.at = (cx, cz) => { const ii = (cx / 4) | 0, kx = cx % 4, o = (cz / 4) | 0, ky = cz % 4; if (kx === 3 || ky === 3 || ii >= I || o >= O) return -1; return ((o * I + ii) * 3 + ky) * 3 + kx; };
        st.describe = (i) => { const kx = i % 3, ky = ((i / 3) | 0) % 3, ii = ((i / 9) | 0) % I, o = (i / (9 * I)) | 0; return { o, ii, ky, kx, text: `filter ${o + 1} of ${O}, looking at input map ${ii + 1} of ${I}, kernel position (${ky + 1}, ${kx + 1})` }; };
      } else {
        const [O, I] = l.shape; st.cols = I; st.rows = O;
        st.cs = Math.min(1, 80 / Math.max(st.cols, st.rows));
        st.cell = (i) => [i % I, (i / I) | 0];
        st.at = (cx, cz) => (cx < I && cz < O ? cz * I + cx : -1);
        st.describe = (i) => { const o = (i / I) | 0, ii = i % I; return { o, ii, text: name === "fc2" ? `row “${d.classes[o]}” (output ${o + 1}), column ${ii + 1}: neuron ${ii + 1} of fc1` : `row ${o + 1} (neuron ${o + 1} of ${O}), column ${ii + 1} of ${I}` }; };
      }
      st.hmax = 10 * Math.sqrt(st.cs);
    } else if (kind === "a") {
      const a = A[name]; st.act = a; st.shape = a.shape;
      if (a.shape.length === 3) gridMaps(st, a.shape[0], a.shape[1], a.shape[2], 1);
      else { const C = a.shape[0], cols = 16; st.mapCols = cols; st.cols = cols; st.rows = Math.ceil(C / cols); st.cs = 2; st.n = C; st.cell = (i) => [i % cols, (i / cols) | 0]; st.at = (cx, cz) => { const i = cz * cols + cx; return cx < cols && i < C ? i : -1; }; st.describe = (i) => ({ text: `neuron ${i + 1} of ${C}` }); st.hmax = 9; }
    } else if (kind === "out") {
      const C = A.fc2.shape[0]; st.act = A.fc2; st.cols = C * 2 - 1; st.rows = 2; st.cs = 3; st.n = C; st.hmax = 22;
      st.cell = (i) => [i * 2, 0]; st.at = (cx, cz) => (cx % 2 === 0 && cx / 2 < C && cz < 2 ? cx / 2 : -1); st.describe = (i) => ({ text: `“${d.classes[i]}”` });
    }
    st.w = st.cols * st.cs; st.d = st.rows * st.cs; st.x0 = x; st.z0 = -st.d / 2; st.cx = x + st.w / 2;
    x += st.w + GAP;
    makeMesh(st);
    S.stages.push(st);
  }
  S.length = x - GAP;
  buildLabels(d);
  buildNav();
}
function gridMaps(st, C, H, W, scale) {
  const cols = Math.ceil(Math.sqrt(C)), rows = Math.ceil(C / cols), tw = W + 1, th = H + 1;
  st.cols = cols * tw - 1; st.rows = rows * th - 1; st.n = C * H * W;
  st.cs = Math.min(scale, 56 / Math.max(st.cols, st.rows));
  st.cell = (i) => { const c = (i / (H * W)) | 0, r = i % (H * W), py = (r / W) | 0, px = r % W; return [(c % cols) * tw + px, ((c / cols) | 0) * th + py]; };
  st.at = (cx, cz) => { const col = (cx / tw) | 0, row = (cz / th) | 0, px = cx % tw, py = cz % th, c = row * cols + col; if (px >= W || py >= H || col >= cols || c >= C) return -1; return c * H * W + py * W + px; };
  st.describe = (i) => { const c = (i / (H * W)) | 0, r = i % (H * W); return { c, text: C > 1 ? `map ${c + 1} of ${C}, pixel (${((r / W) | 0) + 1}, ${(r % W) + 1})` : `pixel (${((r / W) | 0) + 1}, ${(r % W) + 1})` }; };
  st.hmax = 6 * Math.sqrt(st.cs);
}
function makeMesh(st) {
  const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const m = new THREE.InstancedMesh(box, mat, st.n);
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(st.n * 3), 3);
  m.frustumCulled = false;
  st.mesh = m;
  st.px = new Float32Array(st.n); st.pz = new Float32Array(st.n);
  const bw = st.cs * (st.kind === "out" ? 1.6 : 0.86);
  st.bw = bw;
  for (let i = 0; i < st.n; i++) { const [cx, cz] = st.cell(i); st.px[i] = st.x0 + (cx + 0.5) * st.cs; st.pz[i] = st.z0 + (cz + 0.5) * st.cs; }
  st.h = new Float32Array(st.n);
  writeHeights(st, st.h);
  scene.add(m);
  const fl = new THREE.Mesh(new THREE.PlaneGeometry(st.w + 2 * st.cs, st.d + 2 * st.cs), new THREE.MeshBasicMaterial({ color: st.kind === "w" ? P.floorW : P.floor }));
  fl.rotation.x = -Math.PI / 2; fl.position.set(st.cx, -0.03, 0);
  st.floor = fl; scene.add(fl);
}
function writeHeights(st, h) {
  const a = st.mesh.instanceMatrix.array, bw = st.bw;
  for (let i = 0; i < st.n; i++) {
    let v = h[i];
    const neg = v < 0; v = Math.max(Math.abs(v), 0.03);
    const o = i * 16;
    a[o] = bw; a[o + 1] = 0; a[o + 2] = 0; a[o + 3] = 0;
    a[o + 4] = 0; a[o + 5] = v; a[o + 6] = 0; a[o + 7] = 0;
    a[o + 8] = 0; a[o + 9] = 0; a[o + 10] = bw; a[o + 11] = 0;
    a[o + 12] = st.px[i]; a[o + 13] = neg ? -v : 0; a[o + 14] = st.pz[i]; a[o + 15] = 1;   // negative weights hang below the floor
  }
  st.mesh.instanceMatrix.needsUpdate = true;
  st.h = h;
  dirty = true;
}

// ------------------------------------------------------------------ colours and heights from data
function stageValues(st) {
  if (st.kind === "w") {
    const L = S.layers[st.name];
    return L;
  }
  return null;
}
function targetHeights(st, extra = 0) {
  const out = new Float32Array(st.n);
  if (st.kind === "w") {
    const L = S.layers[st.name], w = L.w, m = L.mw;
    const d = L.d;
    for (let i = 0; i < st.n; i++) out[i] = ((w[i] + (d && extra ? d[i] * extra : 0)) / m) * st.hmax;
  } else {
    const v = actValues(st);
    for (let i = 0; i < st.n; i++) out[i] = v[i] * st.hmax;
  }
  return out;
}
function actValues(st) {
  if (st.kind === "in") return S.input;
  const a = S.acts[st.name];
  return a.norm;
}
function lerpColor(out, o, c1, c2, t) { out[o] = c1.r + (c2.r - c1.r) * t; out[o + 1] = c1.g + (c2.g - c1.g) * t; out[o + 2] = c1.b + (c2.b - c1.b) * t; }
function diverge(out, o, v) { const t = Math.min(1, Math.abs(v)) ** 0.6; lerpColor(out, o, P.mid, v < 0 ? P.neg : P.pos, t); }
function heat(out, o, v) { const t = Math.min(1, v); if (t < 0.6) lerpColor(out, o, P.mid, P.hot, (t / 0.6) ** 0.8); else lerpColor(out, o, P.hot, P.hot2, (t - 0.6) / 0.4); }
function colorStage(st, mode = S.mode) {
  const c = st.mesh.instanceColor.array;
  if (st.kind === "w") {
    const L = S.layers[st.name];
    if (mode === "value") for (let i = 0; i < st.n; i++) diverge(c, i * 3, L.w[i] / L.mw);
    else if (mode === "use") for (let i = 0; i < st.n; i++) heat(c, i * 3, Math.sqrt(L.use[i]));
    else if (mode === "grad" && L.g) for (let i = 0; i < st.n; i++) diverge(c, i * 3, -L.g[i] / L.mg);
    else if (mode === "update" && L.d) for (let i = 0; i < st.n; i++) diverge(c, i * 3, L.d[i] / L.md);
    else for (let i = 0; i < st.n; i++) lerpColor(c, i * 3, P.mid, P.mid, 0);
  } else {
    const v = actValues(st);
    for (let i = 0; i < st.n; i++) heat(c, i * 3, v[i]);
  }
  st.mesh.instanceColor.needsUpdate = true;
  dirty = true;
}
function recolorAll() { if (S.built) S.stages.forEach((s) => colorStage(s)); legend(); }

function ingest(d) {
  S.data = d;
  for (const l of d.layers) {
    const w = f16(l.w), g = f16(l.g), dd = l.d ? f16(l.d) : null, use = u8(l.use);
    S.layers[l.name] = { ...l, w, w0: w.slice(), g, d: dd, use, mw: maxAbs(w), mg: g ? maxAbs(g) : 1, md: dd ? maxAbs(dd) : 1 };
  }
  for (const a of d.acts) { const v = u8(a.data, a.max); const m = maxAbs(v); S.acts[a.name] = { ...a, v, norm: v.map((x) => x / m) }; }
  S.input = u8(d.input);
  if (d.history) S.history = d.history;
}

// ------------------------------------------------------------------ labels, legend, caption
let labelEls = [];
function buildLabels(d) {
  const host = $("#in-labels");
  host.innerHTML = "";
  labelEls = S.stages.filter((s) => !(s.kind === "a" && s.name.startsWith("pool") && false)).map((s) => {
    const el = document.createElement("div");
    const title = s.kind === "in" ? "your mark" : s.kind === "w" ? `${s.name} weights` : s.kind === "out" ? "answer" : s.name.startsWith("pool") ? `${s.name}` : `${s.name} output`;
    const sub = s.kind === "w" ? `${fmtInt(s.n)} weights · ${s.shape.join("×")}` : s.kind === "in" ? "28×28 pixels" : s.kind === "out" ? "22 symbols" : `${s.shape.join("×")}`;
    el.className = s.kind === "w" ? "w" : "";
    el.innerHTML = `${esc(title)}<small>${esc(sub)}</small>`;
    host.appendChild(el);
    return { el, s, p: new THREE.Vector3(s.cx, s.kind === "w" ? s.hmax + 3 : s.hmax + 3, s.z0 - 2) };
  });
  // the 22 answer letters
  const out = S.stages.find((s) => s.kind === "out");
  d.classes.forEach((c, i) => {
    const el = document.createElement("div");
    el.style.background = "transparent"; el.style.padding = "0"; el.style.transform = "translate(-50%, 0)";
    el.textContent = c;
    host.appendChild(el);
    labelEls.push({ el, s: out, p: new THREE.Vector3(out.px[i], 0, out.z0 + out.d + 2), letter: i });
  });
}
const _v = new THREE.Vector3();
function placeLabels() {
  const c = $("#in-canvas"), W = c.clientWidth, H = c.clientHeight;
  const far = camera.position.distanceTo(controls.target);
  for (const L of labelEls) {
    _v.copy(L.p).project(camera);
    const on = _v.z < 1 && _v.x > -1.1 && _v.x < 1.1 && _v.y > -1.1 && _v.y < 1.1;
    const tooFar = (L.letter != null && far > 260) || (L.s.kind === "a" && camera.position.distanceTo(L.p) > 260);
    L.el.style.display = on && !tooFar ? "" : "none";
    if (on) { L.el.style.left = `${((_v.x + 1) / 2) * W}px`; L.el.style.top = `${((1 - _v.y) / 2) * H}px`; }
    L.el.classList.toggle("hot", !!S.phaseStage && S.phaseStage === L.s && L.letter == null);
  }
}
const HINTS = {
  value: "Height and colour show each weight's value: orange positive, blue negative, grey near zero. This is the whole of what the network knows.",
  use: "How much each weight mattered for this mark: |weight × the number it multiplied|. Bright weights did the work; dark ones were multiplied by zero this time.",
  grad: "Which way each weight should move to make the right answer more likely: orange up, blue down, grey = no say. Weights that weren't used get no gradient.",
  update: "How far each weight moved in the last step (shown enlarged so you can see it). Every weight that had a gradient moved a little.",
};
function legend() {
  $("#in-mode-hint").textContent = HINTS[S.mode];
  const div = `linear-gradient(90deg, #${P.neg.getHexString()}, #${P.mid.getHexString()}, #${P.pos.getHexString()})`;
  const ht = `linear-gradient(90deg, #${P.mid.getHexString()}, #${P.hot.getHexString()}, #${P.hot2.getHexString()})`;
  const t = { value: ["weights", div, "negative", "positive"], use: ["weights", ht, "unused", "used most"], grad: ["weights", div, "should go down", "should go up"], update: ["weights", div, "moved down", "moved up"] }[S.mode];
  $("#in-legend").innerHTML = `<div>${t[0]}</div><div class="ramp" style="background:${t[1]}"></div><div class="ends"><span>${t[2]}</span><span>${t[3]}</span></div>
    <div style="margin-top:6px">activations</div><div class="ramp" style="background:${ht}"></div><div class="ends"><span>0</span><span>strongest</span></div>`;
}
const cap = (h) => ($("#in-cap").innerHTML = h);

// ------------------------------------------------------------------ hover and pick
const _ndc = new THREE.Vector2(), _hit = new THREE.Vector3();
function locate(e) {
  const r = e.target.getBoundingClientRect();
  _ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(_ndc, camera);
  if (!raycaster.ray.intersectPlane(floorPlane, _hit)) return null;
  const st = S.stages.find((s) => _hit.x >= s.x0 && _hit.x < s.x0 + s.w && _hit.z >= s.z0 && _hit.z < s.z0 + s.d);
  if (!st) return null;
  const cx = Math.floor((_hit.x - st.x0) / st.cs), cz = Math.floor((_hit.z - st.z0) / st.cs);
  const i = st.at(cx, cz);
  return i < 0 ? null : { st, i, r, e };
}
function hover(e) {
  if (!S.built) return;
  const h = locate(e), tip = $("#in-tip");
  if (!h) { tip.hidden = true; dirty = true; if (!S.pinned) marker.visible = false; else showMarker(S.pinned.st, S.pinned.i); return; }
  showMarker(h.st, h.i);
  tip.hidden = false;
  tip.innerHTML = describe(h.st, h.i);
  const x = e.clientX - h.r.left, y = e.clientY - h.r.top;
  tip.style.left = `${Math.min(x + 14, h.r.width - tip.offsetWidth - 6)}px`;
  tip.style.top = `${Math.min(y + 14, h.r.height - tip.offsetHeight - 6)}px`;
}
function showMarker(st, i) {
  const hgt = Math.abs(st.h[i]) + 0.2;
  marker.visible = true;
  marker.scale.set(st.bw * 1.25, hgt, st.bw * 1.25);
  marker.position.set(st.px[i], st.h[i] < 0 ? -hgt : 0, st.pz[i]);
  marker.material.color.set(P.dark ? 0xffffff : 0x111111);
  dirty = true;
}
function inputOf(name) {
  if (name === "fc1") return S.acts.pool3?.v;
  if (name === "fc2") return S.acts.fc1?.v;
  return null;
}
function describe(st, i) {
  const D = st.describe(i);
  if (st.kind === "w") {
    const L = S.layers[st.name], w = L.w[i];
    let h = `<b>${esc(st.name)} · weight ${fmtInt(i + 1)} of ${fmtInt(st.n)}</b>${esc(D.text)}<br>value <code>${sgn(w)}</code>`;
    const inp = inputOf(st.name);
    if (inp) h += `<br>× input <code>${sci(inp[D.ii])}</code> = <code>${sgn(w * inp[D.ii])}</code> to ${st.name === "fc2" ? `“${esc(S.data.classes[D.o])}”` : `neuron ${D.o + 1}`}`;
    if (L.g) h += `<br>gradient <code>${sgn(L.g[i], 3)}</code>`;
    if (L.d) h += `<br>moved <code>${sgn(L.d[i], 3)}</code>`;
    return h + `<br><span style="color:var(--ds-dim)">click to follow this weight</span>`;
  }
  if (st.kind === "out") return `<b>Answer ${esc(D.text)}</b>probability ${pct(S.acts.fc2.v[i])}`;
  const v = st.kind === "in" ? S.input[i] : S.acts[st.name].v[i];
  return `<b>${esc(st.kind === "in" ? "your mark" : st.name)}</b>${esc(D.text)}<br>value <code>${sci(v)}</code>`;
}
function pick(e) {
  const h = locate(e);
  if (!h || h.st.kind !== "w") return;
  S.pinned = { st: h.st, i: h.i, name: h.st.name };
  renderBody();
  document.getElementById("one-weight")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ------------------------------------------------------------------ camera
function flyTo(st) {
  let target, dist;
  let dir;
  if (!st) {
    resize();
    // look along the network from the front left, then back off until both ends are in view
    dir = camera.aspect < 0.9 ? new THREE.Vector3(-0.8, 0.5, 0.33).normalize() : new THREE.Vector3(-0.55, 0.58, 0.6).normalize();
    target = new THREE.Vector3(S.length * 0.42, 0, 0);
    const pts = [];
    for (const s of [S.stages[0], S.stages[S.stages.length - 1]]) for (const z of [s.z0, s.z0 + s.d]) for (const x of [s.x0, s.x0 + s.w]) pts.push(new THREE.Vector3(x, 0, z));
    const cam = camera.clone();
    dist = S.length * 0.25;
    for (let k = 0; k < 60; k++) {
      cam.position.copy(target).addScaledVector(dir, dist); cam.lookAt(target); cam.updateMatrixWorld(); cam.updateProjectionMatrix();
      if (pts.every((p) => { const v = p.clone().project(cam); return Math.abs(v.x) < 0.92 && Math.abs(v.y) < 0.8 && v.z < 1; })) break;
      dist *= 1.08;
    }
  } else { target = new THREE.Vector3(st.cx, 0, 0); dist = Math.max(st.w, st.d, 30) * 1.5 + 20; dir = new THREE.Vector3(-0.32, 0.62, 0.72).normalize(); }
  const from = camera.position.clone(), fromT = controls.target.clone(), to = target.clone().addScaledVector(dir, dist);
  tween(900, (p) => { const k = ease(p); camera.position.lerpVectors(from, to, k); controls.target.lerpVectors(fromT, target, k); });
  $$("#in-stages button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.fly === (st ? st.name + st.kind : "all"))));
}
function buildNav() {
  const ul = $("#in-stages");
  ul.innerHTML = `<li><button data-fly="all" aria-selected="true">Whole network</button></li>` + S.stages.filter((s) => s.kind === "w" || s.kind === "in" || s.kind === "out")
    .map((s) => `<li><button data-fly="${s.name + s.kind}">${s.kind === "in" ? "Your mark" : s.kind === "out" ? "The answer" : `${s.name} <span style="color:var(--ds-faint)">${fmtInt(s.n)}</span>`}</button></li>`).join("");
  $$("#in-stages button").forEach((b) => (b.onclick = () => flyTo(b.dataset.fly === "all" ? null : S.stages.find((s) => s.name + s.kind === b.dataset.fly))));
}

// ------------------------------------------------------------------ the three phases
function setMode(m) {
  S.mode = m;
  $$("#in-mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.cmode === m)));
  recolorAll();
}
$$("#in-mode button").forEach((b) => (b.onclick = () => setMode(b.dataset.cmode)));

function sweep(dir, dur, onStage) {
  return new Promise((res) => {
    const x0 = dir > 0 ? -12 : S.length + 12, x1 = dir > 0 ? S.length + 12 : -12;
    pulse.visible = true;
    pulse.scale.set(200, 40, 1);
    pulse.material.color.set(dir > 0 ? 0xe0a03a : 0xd0453f);
    const order = dir > 0 ? S.stages.slice() : S.stages.slice().reverse();
    let next = 0;
    tween(dur, (p) => {
      const x = x0 + (x1 - x0) * p;
      pulse.position.set(x, 18, 0);
      while (next < order.length && (dir > 0 ? x >= order[next].x0 : x <= order[next].x0 + order[next].w)) { S.phaseStage = order[next]; onStage(order[next]); next++; }
    }, () => { pulse.visible = false; S.phaseStage = null; res(); });
  });
}
function growTo(st, target, dur = 380) {
  const from = st.h.slice();
  tween(dur, (p) => { const k = ease(p), h = new Float32Array(st.n); for (let i = 0; i < st.n; i++) h[i] = from[i] + (target[i] - from[i]) * k; writeHeights(st, h); });
}
async function animForward() {
  const prev = S.mode;
  S.mode = "use"; $$("#in-mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.cmode === "use"))); legend();
  S.stages.forEach((s) => { if (s.kind !== "w") { writeHeights(s, new Float32Array(s.n)); colorStage(s); } });
  cap("<b>Forward pass.</b> Each layer multiplies what comes in by its weights, adds them up and passes the result on. Bright weights are the ones doing the work for this mark.");
  await sweep(1, 3200, (st) => { if (st.kind === "w") colorStage(st, "use"); else { colorStage(st); growTo(st, targetHeights(st)); } });
  const top = topk(S.acts.fc2.v, 1)[0];
  cap(`<b>Forward pass done:</b> it reads the mark as <b>“${esc(S.data.classes[top])}”</b> (${pct(S.acts.fc2.v[top], 0)}). ${fmtInt(S.data.counts.shown)} weights were each used once: about ${fmtInt(flops())} multiply-adds.`);
  if (prev !== "use") { /* stay on use, it is what forward means */ }
}
async function animBackward() {
  S.mode = "grad"; $$("#in-mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.cmode === "grad"))); legend();
  S.stages.forEach((s) => { if (s.kind === "w") colorStage(s, "none"); });
  const lab = $("#in-label").value;
  cap(`<b>Backward pass.</b> Starting from how wrong the answer was (loss ${sci(S.data.loss)} for “${esc(lab)}”), the chain rule works back through every layer and gives each weight its gradient.`);
  await sweep(-1, 3000, (st) => { if (st.kind === "w") colorStage(st, "grad"); });
  const g = S.data.layers.map((l) => `${l.name} ${sci(l.stats.mean_abs_g, 1)}`).join(" · ");
  cap(`<b>Every weight now has a gradient.</b> Average size by layer: ${esc(g)}. Grey weights got none: they weren't used, so changing them couldn't help.`);
}
async function animUpdate() {
  S.mode = "update"; $$("#in-mode button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.cmode === "update"))); legend();
  const opt = S.data.cfg.optimizer, lr = S.data.cfg.lr;
  cap(`<b>Update.</b> ${opt === "sgd" ? `Each weight moves against its gradient: <code>new = old − ${lr} × gradient</code>` : "Adam moves each weight by its gradient scaled by that weight's own recent history"}. Shown ×${fmtInt(S.exag)} so you can see it.`);
  const ws = S.stages.filter((s) => s.kind === "w");
  ws.forEach((s) => colorStage(s, "update"));
  // grow to the exaggerated new value, hold, then settle on the true new value
  ws.forEach((s) => growTo(s, targetHeights(s, S.exag), 900));
  await new Promise((r) => setTimeout(r, 1700));
  // fold the step in: from here on the "current" weights are the new ones
  ws.forEach((s) => { const L = S.layers[s.name]; for (let i = 0; i < s.n; i++) L.w[i] += L.d[i]; L.mw = maxAbs(L.w); });
  ws.forEach((s) => growTo(s, targetHeights(s), 700));
  const lab = $("#in-label").value, k = S.data.classes.indexOf(lab);
  const before = S.data.probs[k], after = S.data.probs_after?.[k];
  const moved = S.data.layers.reduce((a, l) => a + l.stats.changed * l.stats.n, 0);
  cap(`<b>One step done.</b> ${fmtInt(moved)} of ${fmtInt(S.data.counts.shown)} weights moved, by ${sci(avgMoved(), 1)} on average. “${esc(lab)}” went from ${pct(before)} to <b>${pct(after)}</b>.`);
}
const avgMoved = () => { let s = 0, n = 0; S.data.layers.forEach((l) => { s += l.stats.mean_abs_d * l.stats.n; n += l.stats.n; }); return s / n; };
function flops() { let f = 0; for (const l of S.data.layers) { if (l.kind === "fc") f += l.stats.n; else { const a = S.acts[l.name]; f += l.stats.n * a.shape[1] * a.shape[2]; } } return f; }
const topk = (a, k) => [...a.keys()].sort((x, y) => a[y] - a[x]).slice(0, k);

// ------------------------------------------------------------------ actions
function inputBody() { return S.strokes.length ? { strokes: S.strokes } : S.sample ? { sample: S.sample.c, seed: S.sample.seed } : {}; }
async function guard(fn) {
  if (S.busy) return;
  S.busy = true;
  $$(".in-steps .ds-btn, #in-all").forEach((b) => (b.disabled = true));
  try { await fn(); } catch (e) { cap(esc(e.message)); DocsShell.toast(e.message); }
  S.busy = false;
  $$(".in-steps .ds-btn, #in-all").forEach((b) => (b.disabled = false));
}
function first(d) {
  ingest(d);
  if (!S.built) { buildStages(d); S.built = true; S.stages.forEach((s) => { writeHeights(s, targetHeights(s)); colorStage(s); }); flyTo(null); legend(); }
  if (!S.labelsFilled) { $("#in-label").innerHTML = d.classes.map((c) => `<option>${esc(c)}</option>`).join(""); S.labelsFilled = true; }
  if (!S.labelTouched) { const top = topk(S.acts.fc2.v, 1)[0]; $("#in-label").value = S.sample ? S.sample.c : d.classes[top]; }
  drawInput();
  renderBody();
}
async function forward(animate = true) {
  const d = await api("/api/inside/look", inputBody());
  const was = S.built;
  first(d);
  steps(1);
  if (animate && was) await animForward();
  else { setMode(S.mode); cap("Press <b>1 · Forward</b> to watch the mark go through, or <b>One full training step</b> to see all three phases."); }
}
async function backward() {
  const d = await api("/api/inside/look", { ...inputBody(), label: $("#in-label").value });
  first(d); steps(2);
  await animBackward();
}
async function update() {
  const d = await api("/api/inside/look", { ...inputBody(), label: $("#in-label").value, step: true });
  first(d); steps(3);
  await animUpdate();
  renderBody();
}
async function fullStep() {
  const d = await api("/api/inside/look", { ...inputBody(), label: $("#in-label").value, step: true });
  first(d);
  steps(1); await animForward();
  steps(2); await animBackward();
  steps(3); await animUpdate();
  renderBody();
}
function steps(n) { $$(".in-steps .ds-btn").forEach((b, i) => b.classList.toggle("done", i < n)); }
$("#in-fwd").onclick = () => guard(() => forward(true));
$("#in-bwd").onclick = () => guard(backward);
$("#in-upd").onclick = () => guard(update);
$("#in-all").onclick = () => guard(fullStep);
$("#in-label").onchange = () => { S.labelTouched = true; steps(1); };

// training on batches, live
async function playLoop() {
  while (S.playing) {
    try {
      const t = await api("/api/inside/train", { steps: 10, batch: 32 });
      const d = await api("/api/inside/look", inputBody());
      const old = {};
      S.stages.forEach((s) => (old[s.name + s.kind] = s.h.slice()));
      ingest(d);
      S.history = t.history;
      S.stages.forEach((s) => { colorStage(s, S.mode === "grad" || S.mode === "update" ? "value" : S.mode); growTo(s, targetHeights(s), 450); });
      if (S.mode === "grad" || S.mode === "update") setMode("value");
      const top = topk(S.acts.fc2.v, 1)[0];
      cap(`<b>Training on batches of 32.</b> Step ${fmtInt(t.steps)} · loss ${t.loss.toFixed(3)} · batch accuracy ${pct(t.acc, 0)} · your mark now reads as “${esc(d.classes[top])}” (${pct(S.acts.fc2.v[top], 0)}).`);
      S.data = d;
      drawInput();
      renderBody(true);
    } catch (e) { DocsShell.toast(e.message); S.playing = false; }
  }
  $("#in-play").textContent = "▶ Train on batches";
}
$("#in-play").onclick = () => {
  if (S.busy && !S.playing) return;
  S.playing = !S.playing;
  $("#in-play").textContent = S.playing ? "❚❚ Pause" : "▶ Train on batches";
  if (S.playing) playLoop();
};
async function reset(start) {
  S.playing = false;
  const r = await api("/api/inside/reset", { start, optimizer: S.opt });
  $("#in-lr").value = r.cfg.lr;
  S.history = [];
  await guard(() => forward(false));
  S.stages.forEach((s) => { writeHeights(s, targetHeights(s)); colorStage(s); });
  setMode("value");
  cap(start === "random" ? "<b>Every weight is random now.</b> The network guesses at chance. Press <b>▶ Train on batches</b> and watch structure appear." : "A fresh copy of the reader the Solve tab uses.");
}
$("#in-reset-t").onclick = () => reset("trained");
$("#in-reset-r").onclick = () => reset("random");
S.opt = "sgd";
$$("#in-opt button").forEach((b) => (b.onclick = async () => {
  S.opt = b.dataset.opt;
  $$("#in-opt button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  const r = await api("/api/inside/optimizer", { optimizer: S.opt });
  if (S.data) S.data.cfg = r.cfg;
}));
$("#in-lr").onchange = async () => { try { const r = await api("/api/inside/optimizer", { lr: +$("#in-lr").value }); if (S.data) S.data.cfg = r.cfg; } catch (e) { DocsShell.toast(e.message); } };
const exag = $("#in-exag");
exag.oninput = () => { S.exag = Math.round(10 ** +exag.value); $("#in-exag-v").textContent = fmtInt(S.exag); };
exag.oninput();

// ------------------------------------------------------------------ input pad
const pad = $("#in-pad"), pctx = pad.getContext("2d");
let cur = null;
function drawInput() {
  pctx.setTransform(1, 0, 0, 1, 0, 0);
  pctx.fillStyle = "#000"; pctx.fillRect(0, 0, 112, 112);
  if (S.strokes.length || cur) {
    pctx.strokeStyle = "#fff"; pctx.lineWidth = 7; pctx.lineCap = pctx.lineJoin = "round";
    for (const s of S.strokes.concat(cur ? [cur] : [])) { pctx.beginPath(); s.forEach((p, i) => (i ? pctx.lineTo(p[0], p[1]) : pctx.moveTo(p[0], p[1]))); if (s.length === 1) pctx.lineTo(s[0][0] + 0.1, s[0][1]); pctx.stroke(); }
  } else if (S.input) {
    const img = pctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) { const v = Math.round(S.input[i] * 255); img.data.set([v, v, v, 255], i * 4); }
    const t = document.createElement("canvas"); t.width = t.height = 28; t.getContext("2d").putImageData(img, 0, 0);
    pctx.imageSmoothingEnabled = false; pctx.drawImage(t, 0, 0, 112, 112);
  }
}
const ppos = (e) => { const r = pad.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * 112, ((e.clientY - r.top) / r.height) * 112]; };
pad.addEventListener("pointerdown", (e) => { e.preventDefault(); pad.setPointerCapture(e.pointerId); if (S.sample) { S.sample = null; S.strokes = []; } cur = [ppos(e)]; drawInput(); });
pad.addEventListener("pointermove", (e) => { if (!cur) return; cur.push(ppos(e)); drawInput(); });
const endStroke = () => { if (!cur) return; S.strokes.push(cur); cur = null; S.labelTouched = false; clearTimeout(S.t); S.t = setTimeout(() => guard(() => forward(false)), 350); };
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);
$("#in-clear").onclick = () => { S.strokes = []; S.sample = null; drawInput(); };
const SAMPLES = ["7", "1", "b", "6", "2", "z", "x", "+", "4", "9", "="];
$("#in-samples").innerHTML = SAMPLES.map((c) => `<button data-s="${esc(c)}" title="A synthetic ${esc(c)}">${esc(c)}</button>`).join("");
$$("#in-samples button").forEach((b) => (b.onclick = () => { S.strokes = []; S.sample = { c: b.dataset.s, seed: Math.floor(Math.random() * 1e6) }; S.labelTouched = false; guard(() => forward(S.built)); }));

// ------------------------------------------------------------------ the page under the scene
function renderBody(light = false) {
  const d = S.data;
  if (!d) return;
  $("#in-meta").textContent = `${fmtInt(d.counts.total)} parameters: ${fmtInt(d.counts.shown)} weights, each one a block below, plus ${fmtInt(d.counts.other)} biases and normalisation numbers · started from ${d.source} · ${fmtInt(d.steps)} training steps taken here`;
  let h = "";
  // what it said
  const k = topk(d.probs, 5);
  const after = d.probs_after;
  h += `<h2 id="what-it-said">What it said</h2><p class="ds-sub">Probabilities for the mark on the left${after ? ", before and after the last step" : ""}.</p><div class="in-probs">${after ? `<span></span><span class="h">before</span><span></span><span class="h">after one step</span><span></span>` : ""}`;
  for (const i of k) h += `<b>${esc(d.classes[i])}</b><span class="b"><i style="width:${d.probs[i] * 100}%"></i></span><small>${pct(d.probs[i])}</small>${after ? `<span class="b after"><i style="width:${after[i] * 100}%"></i></span><small>${pct(after[i])}</small>` : "<span></span><span></span>"}`;
  h += `</div>`;
  // layer table
  h += `<h2 id="layer-by-layer">Layer by layer</h2><p class="ds-sub">Where the weights are, how much of this mark's work each layer did, and what the last step did to them.</p><div class="tablewrap"><table><thead><tr><th>Layer</th><th class="num">Weights</th><th class="num">Share of all</th><th class="num">Share of the work</th><th class="num">Average |weight|</th><th class="num">Average |gradient|</th><th class="num">Average move</th><th class="num">Weights moved</th></tr></thead><tbody>`;
  for (const l of d.layers) {
    const s = l.stats;
    h += `<tr><td><b>${esc(l.name)}</b> <small class="ds-hint">${l.shape.join("×")}</small><br><small class="ds-hint">${esc(l.what)}</small></td><td class="num">${fmtInt(s.n)}</td><td class="num">${pct(s.n / d.counts.shown)}</td><td class="num">${pct(s.use_share)}</td><td class="num">${sci(s.mean_abs_w, 2)}</td><td class="num">${s.mean_abs_g != null ? sci(s.mean_abs_g, 1) : "–"}</td><td class="num">${s.mean_abs_d != null ? sci(s.mean_abs_d, 1) : "–"}</td><td class="num">${s.changed != null ? pct(s.changed, 0) : "–"}</td></tr>`;
  }
  h += `</tbody></table></div><p class="ds-hint">fc1 holds over half of all the weights but, for any one mark, most of its neurons are switched off (zero after ReLU), so most of its weights get no gradient and don't move. The convolution weights are few but used at every pixel, ${fmtInt(flops())} multiply-adds in all.</p>`;
  // one weight
  h += `<h2 id="one-weight">One weight, followed</h2>`;
  if (S.pinned) {
    const L = S.layers[S.pinned.name], i = S.pinned.i, st = S.stages.find((s) => s.name === S.pinned.name && s.kind === "w"), D = st.describe(i);
    const w = L.w0[i], g = L.g?.[i], dd = L.d?.[i];
    const lr = d.cfg.lr;
    h += `<div class="in-pin"><p style="margin-top:0"><b>${esc(S.pinned.name)}, weight ${fmtInt(i + 1)}</b>: ${esc(D.text)}.</p><div class="eq">`;
    const inp = inputOf(S.pinned.name);
    h += `value    = ${sgn(w, 6)}<br>`;
    if (inp) h += `forward:  ${sgn(w, 4)} × input ${sci(inp[D.ii], 4)} = ${sgn(w * inp[D.ii], 4)}  → added into ${S.pinned.name === "fc2" ? `“${esc(d.classes[D.o])}”` : `neuron ${D.o + 1}`}<br>`;
    else h += `forward:  multiplied by every ${S.pinned.name === "conv1" ? "pixel" : "position"} of input map ${D.ii + 1}, for filter ${D.o + 1}<br>`;
    if (g != null) h += `backward: gradient = ${sgn(g, 4)} (${Math.abs(g) < 1e-9 ? "no say in this answer" : g < 0 ? "raising it would lower the loss" : "lowering it would lower the loss"})<br>`;
    if (dd != null) h += d.cfg.optimizer === "sgd" ? `update:   new = old − ${lr} × gradient = ${sgn(w, 6)} − ${lr} × ${sgn(g ?? 0, 4)} = ${sgn(w + dd, 6)}` + (Math.abs(dd + lr * (g ?? 0)) > 1e-6 ? ` (gradients were clipped to length 1)` : "") : `update:   moved ${sgn(dd, 4)} (Adam)`;
    h += `</div></div>`;
  } else h += `<p>Click any block in the scene to follow that one weight: its value, what it was multiplied by on the way forward, its gradient on the way back, and the arithmetic of its update.</p>`;
  // learning curve
  if (S.history.length > 1) h += `<h2 id="learning">Learning</h2><p class="ds-sub">Loss on each training batch here (lower is better).</p><div id="in-curve"></div>`;
  $("#in-body").innerHTML = h;
  if (S.history.length > 1 && window.lineChart) {
    const hs = S.history.filter((x) => x.loss != null);
    const step = Math.max(1, Math.floor(hs.length / 120));
    const pts = hs.filter((_, i) => i % step === 0);
    window.lineChart($("#in-curve"), { title: "loss", x: pts.map((p) => p.step), xName: "step", xLabel: "training step", yFmt: (v) => v.toFixed(2), series: [{ name: "loss", color: css("--s1"), values: pts.map((p) => p.loss) }] });
  }
  DocsShell.refresh();
}

// ------------------------------------------------------------------ scale: from this to 20B
const SHOWN = 140696;
function human(n) { const u = [[1e12, "trillion"], [1e9, "billion"], [1e6, "million"], [1e3, "thousand"]]; for (const [v, w] of u) if (n >= v) return `${(n / v).toFixed(n / v < 10 ? 1 : 0)} ${w}`; return fmtInt(n); }
function bytesH(b) { const u = [[1e15, "PB"], [1e12, "TB"], [1e9, "GB"], [1e6, "MB"], [1e3, "KB"]]; for (const [v, w] of u) if (b >= v) return `${(b / v).toFixed(b / v < 10 ? 1 : 0)} ${w}`; return `${b} B`; }
function dur(s) { const u = [[31557600, "years"], [86400, "days"], [3600, "hours"], [60, "minutes"]]; for (const [v, w] of u) if (s >= v) return `${(s / v).toFixed(s / v < 10 ? 1 : 0)} ${w}`; return `${s.toFixed(1)} seconds`; }
function scaleHtml() {
  return `<div id="scale-wrap"><h2 id="scale">From 141 thousand to 20 billion</h2>
  <p class="ds-sub">The same three things happen to every weight of a 20-billion-parameter model on every training step: it is used in the forward pass, it gets a gradient, it moves a little. Only the count changes. Slide to any size.</p>
  <label class="ds-field">Parameters: <b id="scale-v"></b><input type="range" id="scale-n" min="5" max="12" step="0.01" value="${Math.log10(2e10).toFixed(2)}"></label>
  <div class="scale-out" id="scale-out"></div>
  <h3 id="where-they-live">Where they live in a typical language model of this size</h3><div id="scale-tf"></div>
  <p class="ds-hint">Estimates use the standard rules of thumb: a forward pass costs about 2 operations per parameter per token and a training step about 6 (forward, backward and update); a compute-efficient run sees about 20 tokens per parameter; training with Adam in mixed precision keeps about 16 bytes per parameter (weights, gradients, a full-precision master copy and Adam's two running averages), before activations. The GPU figure assumes 400 trillion operations a second actually achieved.</p></div>`;
}
function wireScale() {
  const r = $("#scale-n");
  if (!r) return;
  const upd = () => {
    const N = 10 ** +r.value;
    $("#scale-v").textContent = `${human(N)} (${fmtInt(N / SHOWN)}× this network)`;
    const train = 6 * N * 20 * N, gpu = 4e14;
    const cards = [
      [bytesH(2 * N), "to store the weights in 16-bit numbers"],
      [bytesH(16 * N), "of memory to train it with Adam, before activations"],
      [`${human(2 * N)}`, "multiply-adds per token on the way forward: each weight used once"],
      [`${human(20 * N)} tokens`, "to train it compute-efficiently (20 per parameter)"],
      [`${train.toExponential(1).replace("e+", " × 10^")}`, "operations for that whole training run"],
      [dur(train / gpu), "on one GPU at 400 trillion operations a second"],
      [dur(train / gpu / 1024), "on 1,024 of them"],
      [`${fmtInt(N / SHOWN)}`, "copies of the scene above to show one block per weight"],
    ];
    $("#scale-out").innerHTML = cards.map(([b, s]) => `<div><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");
    // a transformer with d_model = 128 × layers: 12·d² per layer
    let Lr = Math.max(1, Math.round(Math.cbrt(N / 196608)));
    let dm = Math.max(128, Math.round((128 * Lr) / 128) * 128);
    const vocab = 50000, perLayer = 12 * dm * dm, emb = vocab * dm, tot = perLayer * Lr + emb;
    const rows = [["Attention: query, key, value, output", `4 matrices of ${fmtInt(dm)} × ${fmtInt(dm)}`, 4 * dm * dm], ["Feed-forward (like fc1 and fc2 here)", `${fmtInt(dm)} × ${fmtInt(4 * dm)} and ${fmtInt(4 * dm)} × ${fmtInt(dm)}`, 8 * dm * dm]];
    $("#scale-tf").innerHTML = `<p>About <b>${Lr} layers</b>, each with vectors of <b>${fmtInt(dm)}</b> numbers, gives ${human(tot)} parameters:</p><div class="tablewrap"><table><thead><tr><th>Part</th><th>Shape</th><th class="num">Per layer</th><th class="num">× ${Lr} layers</th><th class="num">Share</th></tr></thead><tbody>${rows.map(([a, b, n]) => `<tr><td>${a}</td><td>${b}</td><td class="num">${human(n)}</td><td class="num">${human(n * Lr)}</td><td class="num">${pct((n * Lr) / tot, 0)}</td></tr>`).join("")}<tr><td>Word embeddings</td><td>${fmtInt(vocab)} words × ${fmtInt(dm)}</td><td class="num">–</td><td class="num">${human(emb)}</td><td class="num">${pct(emb / tot, 0)}</td></tr></tbody></table></div><p>Each of those matrices works the way fc1 does above: every output is one row of weights times the incoming vector, added up. One of these layers alone, ${human(perLayer)} weights, is ${fmtInt(perLayer / SHOWN)} times this whole network.</p>`;
  };
  r.oninput = upd;
  upd();
}

// ------------------------------------------------------------------ start when the tab is first opened
let started = false;
function start() {
  if (started || !visible()) return;
  started = true;
  init3d();
  $("#in-scale").innerHTML = scaleHtml();
  wireScale();
  S.sample = { c: "7", seed: 3 };
  guard(() => forward(false));
}
document.addEventListener("ds:view", (e) => { if (e.detail === "inside") { start(); setTimeout(resize, 50); } });
start();
