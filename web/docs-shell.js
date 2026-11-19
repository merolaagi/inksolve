/* docs-shell.js: the behaviour behind docs-shell.css. No dependencies, no build step.
     DocsShell.init({
       search: { sources: [async (q) => [{ group, title, sub, href } | { group, title, sub, run }]], placeholder },
       routeTabs: true,     // <button data-ds-tab="x"> shows <section data-ds-view="x">; false for link tabs (multi-page apps)
       markdown: true,      // Copy as Markdown / View as Markdown above each .ds-doc that has content
       toc: true,           // "On this page" from the headings of the visible .ds-doc
       key: "ds",           // localStorage prefix (rail width, theme)
     });
   Then DocsShell.refresh() after the app renders new content (a MutationObserver also catches most changes). */
(function () {
  "use strict";
  const root = document.documentElement;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const slug = (t) => String(t).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "section";
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private window */ } },
  };
  const reduce = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  const S = { opts: {}, items: [], built: "" };

  function toast(msg) {
    const t = document.createElement("div");
    t.className = "ds-toast";
    t.setAttribute("role", "status");
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2600);
  }

  // ------------------------------------------------------------ theme: follows the system until chosen
  const SUN = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const MOON = '<svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>';
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
  function theme(tools) {
    const k = `${S.opts.key}-theme`;
    const saved = store.get(k);
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
    if (!tools || $(".ds-theme", tools)) return;
    const b = document.createElement("button");
    b.className = "ds-theme";
    const paint = () => { b.innerHTML = isDark() ? SUN : MOON; b.title = b.ariaLabel = isDark() ? "Switch to light mode" : "Switch to dark mode"; };
    b.onclick = () => { root.dataset.theme = isDark() ? "light" : "dark"; store.set(k, root.dataset.theme); paint(); document.dispatchEvent(new CustomEvent("ds:theme", { detail: root.dataset.theme })); };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paint);
    paint();
    tools.appendChild(b);
  }

  // ------------------------------------------------------------ views and tabs
  const views = () => $$(".ds-view");
  const active = () => views().find((v) => !v.hidden) || views()[0];
  function show(name, push = true) {
    const vs = views();
    if (!vs.some((v) => v.dataset.dsView === name)) return;
    vs.forEach((v) => (v.hidden = v.dataset.dsView !== name));
    $$(".ds-tabs [data-ds-tab]").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.dsTab === name)));
    if (push && location.hash !== `#/${name}`) history.replaceState(null, "", `#/${name}`);
    scrollTo(0, 0);
    refresh();
    document.dispatchEvent(new CustomEvent("ds:view", { detail: name }));
  }

  // ------------------------------------------------------------ the left rail's draggable edge
  const RAIL = { def: 300, min: 220, fold: 140 };
  const maxW = () => Math.max(RAIL.min, Math.min(620, Math.round(innerWidth * 0.45)));
  function setRail(w, keep = true) {
    const off = w === 0;
    root.classList.toggle("ds-rail-off", off);
    const px = off ? 0 : Math.max(RAIL.min, Math.min(maxW(), Math.round(w)));
    root.style.setProperty("--ds-rail-w", `${px}px`);
    $$(".ds-handle").forEach((h) => { h.setAttribute("aria-valuenow", String(px)); h.title = off ? "Show the panel" : "Drag to resize; nearly shut hides it; double-click resets"; });
    if (keep) store.set(`${S.opts.key}-rail`, String(px));
  }
  const railNow = () => (root.classList.contains("ds-rail-off") ? 0 : parseInt(getComputedStyle(root).getPropertyValue("--ds-rail-w"), 10));
  function handle(v, rail) {
    const h = document.createElement("div");
    h.className = "ds-handle";
    h.setAttribute("role", "separator");
    h.setAttribute("aria-orientation", "vertical");
    h.setAttribute("aria-label", "Resize the side panel");
    h.setAttribute("aria-valuemin", "0");
    h.tabIndex = 0;
    h.innerHTML = '<span class="ds-grip"></span>';
    rail.after(h);
    h.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      h.setPointerCapture(e.pointerId);
      h.classList.add("drag");
      document.body.classList.add("ds-dragging");
      const left = v.getBoundingClientRect().left;
      const move = (ev) => setRail(ev.clientX - left < RAIL.fold ? 0 : ev.clientX - left, false);
      const up = (ev) => {
        h.classList.remove("drag");
        document.body.classList.remove("ds-dragging");
        h.removeEventListener("pointermove", move);
        h.removeEventListener("pointerup", up);
        h.removeEventListener("pointercancel", up);
        const w = ev.clientX - left;
        if (Math.abs(ev.clientX - e.clientX) < 3 && root.classList.contains("ds-rail-off")) setRail(RAIL.def);   // click on the folded tab
        else setRail(w < RAIL.fold ? 0 : w);
      };
      h.addEventListener("pointermove", move);
      h.addEventListener("pointerup", up);
      h.addEventListener("pointercancel", up);
    });
    h.addEventListener("dblclick", () => setRail(RAIL.def));
    h.addEventListener("keydown", (e) => {
      const cur = railNow(), step = e.shiftKey ? 64 : 16;
      const k = { ArrowLeft: () => setRail(cur - step < RAIL.fold ? 0 : cur - step), ArrowRight: () => setRail(cur === 0 ? RAIL.min : cur + step),
        Home: () => setRail(0), End: () => setRail(maxW()), Enter: () => setRail(cur === 0 ? RAIL.def : 0) }[e.key];
      if (k) { e.preventDefault(); k(); }
    });
  }

  // ------------------------------------------------------------ "On this page"
  function headings(v) {
    const doc = $(".ds-doc", v);
    return doc ? $$("h1, h2, h3", doc).filter((h) => h.offsetParent !== null && h.textContent.trim() && !h.closest(".ds-no-toc")) : [];
  }
  function buildToc() {
    const v = active();
    if (!v) return;
    tools(v);
    const toc = $(".ds-toc", v);
    if (!toc || S.opts.toc === false) return;
    const used = new Set();
    S.items = headings(v).map((h) => {
      if (!h.id || used.has(h.id)) {
        let id = `${v.dataset.dsView || "p"}-${slug(h.textContent)}`, n = 2;
        while (used.has(id) || (document.getElementById(id) && document.getElementById(id) !== h)) id = `${id.replace(/-\d+$/, "")}-${n++}`;
        h.id = id;
      }
      used.add(h.id);
      return { h, id: h.id, text: h.textContent.trim().replace(/\s+/g, " ").slice(0, 90), sub: h.tagName === "H3" };
    });
    const sig = (v.dataset.dsView || "") + S.items.map((x) => x.id).join("|");
    if (sig !== S.built) {
      S.built = sig;
      toc.innerHTML = S.items.length > 1 ? `<p class="ds-toc-t">On this page</p><ul>${S.items.map((x) => `<li class="${x.sub ? "sub" : ""}"><a href="#${x.id}" data-id="${x.id}">${esc(x.text)}</a></li>`).join("")}</ul>` : "";
      $$("a", toc).forEach((a) => (a.onclick = (e) => { e.preventDefault(); jump(a.dataset.id); }));
    }
    spy();
  }
  function jump(id) { const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: reduce() ? "auto" : "smooth", block: "start" }); }
  function spy() {
    const v = active();
    if (!v || !S.items.length) return;
    const line = parseFloat(getComputedStyle(root).getPropertyValue("--ds-hdr")) + 40;
    let cur = S.items[0];
    for (const x of S.items) if (x.h.getBoundingClientRect().top <= line) cur = x;
    if (innerHeight + scrollY >= document.body.scrollHeight - 4) cur = S.items[S.items.length - 1];
    $$(".ds-toc a", v).forEach((a) => a.classList.toggle("on", a.dataset.id === cur.id));
    const navs = $$(".ds-rail .ds-nav a[href^='#']", v).filter((a) => S.items.some((x) => `#${x.id}` === a.getAttribute("href")));
    if (navs.length) navs.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === `#${cur.id}`));   // rail links to headings follow the scroll
  }

  // ------------------------------------------------------------ Copy as Markdown / View as Markdown
  const ICO_COPY = '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>';
  const ICO_MD = '<svg viewBox="0 0 24 24"><rect x="2.5" y="5" width="19" height="14" rx="2"/><path d="M6 15V9l2.5 3L11 9v6M15 9v6m0 0-2-2m2 2 2-2"/></svg>';
  const skip = (n) => ["BUTTON", "INPUT", "SELECT", "TEXTAREA", "SCRIPT", "STYLE", "CANVAS", "TEMPLATE"].includes(n.tagName) ||
    n.classList.contains("ds-doc-tools") || n.classList.contains("ds-no-md") || n.hidden || n.getAttribute("aria-hidden") === "true";
  function inline(n) {
    if (n.nodeType === 3) return n.textContent.replace(/\s+/g, " ");
    if (n.nodeType !== 1 || skip(n)) return "";
    const t = n.tagName, inner = [...n.childNodes].map(inline).join("");
    if (t === "B" || t === "STRONG") return inner.trim() ? `**${inner.trim()}**` : "";
    if (t === "I" || t === "EM") return inner.trim() ? `*${inner.trim()}*` : "";
    if (t === "CODE") return `\`${n.textContent}\``;
    if (t === "BR") return "\n";
    if (t === "A" && /^https?:/.test(n.getAttribute("href") || "")) return `[${inner.trim()}](${n.getAttribute("href")})`;
    return inner;
  }
  function md(node, out) {
    for (const n of node.childNodes) {
      if (n.nodeType === 3) { const t = n.textContent.trim(); if (t) out.push(t); continue; }
      if (n.nodeType !== 1 || skip(n)) continue;
      const t = n.tagName;
      if (/^H[1-6]$/.test(t)) out.push("", `${"#".repeat(+t[1])} ${n.textContent.trim().replace(/\s+/g, " ")}`, "");
      else if (t === "P" || t === "BLOCKQUOTE") { const x = inline(n).trim(); if (x) out.push(t === "BLOCKQUOTE" ? `> ${x}` : x, ""); }
      else if (t === "TABLE") {
        const rows = $$("tr", n).map((r) => [...r.children].map((c) => inline(c).trim().replace(/\|/g, "\\|").replace(/\n/g, " ")));
        if (!rows.length) continue;
        const w = Math.max(...rows.map((r) => r.length)), pad = (r) => [...r, ...Array(w - r.length).fill("")];
        out.push("", `| ${pad(rows[0]).join(" | ")} |`, `| ${Array(w).fill("---").join(" | ")} |`, ...rows.slice(1).map((r) => `| ${pad(r).join(" | ")} |`), "");
      } else if (t === "UL" || t === "OL") { [...n.children].forEach((li, i) => out.push(`${t === "OL" ? `${i + 1}.` : "-"} ${inline(li).trim()}`)); out.push(""); }
      else if (t === "PRE") out.push("", "```", n.textContent.replace(/\n$/, ""), "```", "");
      else if (t.toLowerCase() === "svg") { const l = n.getAttribute("aria-label"); out.push(`*[figure${l ? `: ${l}` : ""}]*`, ""); }
      else if (t === "IMG") out.push(`![${n.alt || ""}](${n.src})`, "");
      else if (t === "SUMMARY") out.push(`**${n.textContent.trim()}**`, "");
      else md(n, out);
    }
    return out;
  }
  function markdown(v = active()) {
    const doc = v && $(".ds-doc", v);
    if (!doc) return "";
    return md(doc, []).join("\n").replace(/\n{3,}/g, "\n\n").trim() + "\n";
  }
  let viewer;
  function viewMd(text, title) {
    if (!viewer) {
      viewer = document.createElement("div");
      viewer.className = "ds-mdview";
      viewer.hidden = true;
      viewer.innerHTML = '<div role="dialog" aria-modal="true"><div class="ds-mvh"><span></span><button class="ds-btn" data-c>Copy</button><button class="ds-btn" data-d>Download .md</button><button class="ds-btn" data-x>Close</button></div><pre></pre></div>';
      document.body.appendChild(viewer);
      viewer.onclick = (e) => { if (e.target === viewer || e.target.closest("[data-x]")) viewer.hidden = true; };
      $("[data-c]", viewer).onclick = () => copy($("pre", viewer).textContent);
      $("[data-d]", viewer).onclick = () => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([$("pre", viewer).textContent], { type: "text/markdown" }));
        a.download = `${slug(viewer.dataset.title || "page")}.md`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      };
      document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !viewer.hidden) viewer.hidden = true; });
    }
    $("pre", viewer).textContent = text;
    viewer.dataset.title = title;
    $(".ds-mvh span", viewer).textContent = `${title} · Markdown`;
    viewer.hidden = false;
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); }
    catch { const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove(); }
    toast("Copied as Markdown");
  }
  function tools(v) {
    if (S.opts.markdown === false) return;
    const doc = $(".ds-doc", v);
    if (!doc) return;
    let bar = $(":scope > .ds-doc-tools", doc);
    // only on a page with content: never make an empty area non-empty (apps often draw an intro when it is empty)
    if (!$("h1, h2, p, table, li", doc) || $(".ds-empty", doc)) { if (bar) bar.remove(); return; }
    if (bar) return;
    bar = document.createElement("div");
    bar.className = "ds-doc-tools";
    bar.innerHTML = `<button type="button" data-ds-copy>${ICO_COPY}Copy as Markdown</button><button type="button" data-ds-viewmd>${ICO_MD}View as Markdown</button>`;
    $("[data-ds-copy]", bar).onclick = () => copy(markdown(v));
    $("[data-ds-viewmd]", bar).onclick = () => { const h = $("h1", doc); viewMd(markdown(v), h ? h.textContent.trim() : document.title); };
    doc.prepend(bar);
  }

  // ------------------------------------------------------------ search: ⌘K or /, dims the page
  function search(top) {
    const box = document.createElement("div");
    box.className = "ds-search";
    box.setAttribute("role", "search");
    const mac = /Mac|iPhone|iPad/.test(navigator.platform);
    box.innerHTML = `<svg class="ds-ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>
      <input type="search" placeholder="${esc((S.opts.search || {}).placeholder || "Search")}" autocomplete="off" spellcheck="false" aria-label="Search" aria-expanded="false">
      <kbd>${mac ? "⌘K" : "Ctrl K"}</kbd><div class="ds-results" role="listbox" hidden></div>`;
    top.insertBefore(box, $(".ds-tools", top) || null);
    const q = $("input", box), out = $(".ds-results", box);
    const scrim = document.createElement("div");
    scrim.className = "ds-scrim";
    document.body.appendChild(scrim);
    const X = { res: [], sel: 0, seq: 0, t: 0 };
    const words = (s) => String(s || "").toLowerCase().match(/[a-z0-9]+/g) || [];
    const local = () => {
      const items = [];
      $$(".ds-tabs > *").forEach((t) => items.push({ group: "Pages", title: t.textContent.trim(), sub: t.dataset.dsHint || "",
        run: () => (t.dataset.dsTab && S.opts.routeTabs !== false ? show(t.dataset.dsTab) : t.click()) }));
      $$(".ds-view").forEach((v) => $$(".ds-nav a, .ds-nav button", v).forEach((a) => items.push({ group: "Pages", title: a.textContent.trim(),
        sub: (($(".ds-nav-title", v) || {}).textContent || "").trim(), run: () => { if (v.hidden && v.dataset.dsView) show(v.dataset.dsView); a.click(); } })));
      S.items.forEach((x) => items.push({ group: "On this page", title: x.text, run: () => jump(x.id) }));
      return items;
    };
    const match = (items, text) => {
      const toks = words(text);
      return items.map((it) => {
        const tw = words(it.title), rw = words(it.sub);
        let sc = 0;
        for (const [i, k] of toks.entries()) {
          const hit = (ws) => ws.some((w) => w === k || ((i === toks.length - 1 || k.length > 3) && w.startsWith(k)));
          if (hit(tw)) sc += 3; else if (hit(rw)) sc += 1; else return null;
        }
        return [sc, it];
      }).filter(Boolean).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
    };
    const hl = (s, text) => {
      const toks = words(text).filter((t) => t.length > 1);
      const h = esc(s || "");
      return toks.length ? h.replace(new RegExp(`\\b(${toks.join("|")})`, "gi"), "<mark>$1</mark>") : h;
    };
    function draw(text, pending) {
      if (!text) { out.innerHTML = '<div class="ds-meta">Type to start searching</div>'; return; }
      if (!X.res.length) { out.innerHTML = `<div class="ds-meta">${pending ? "Searching…" : "No matching results"}</div>`; return; }
      let html = `<div class="ds-meta">${X.res.length}${pending ? "+" : ""} matching result${X.res.length === 1 ? "" : "s"}</div>`, g = null;
      X.res.forEach((r, i) => {
        if (r.group !== g) { g = r.group; html += `<div class="ds-g">${esc(g || "")}</div>`; }
        html += `<div class="ds-r${i === X.sel ? " on" : ""}" role="option" data-i="${i}"><span class="ds-ri" aria-hidden="true">${esc(r.icon || "·")}</span><div><b>${hl(r.title, text)}</b>${r.sub ? `<small>${hl(r.sub, text)}</small>` : ""}</div></div>`;
      });
      out.innerHTML = html;
      $$(".ds-r", out).forEach((el) => {
        el.onmousedown = (e) => e.preventDefault();
        el.onclick = () => go(X.res[+el.dataset.i]);
        el.onmousemove = () => { X.sel = +el.dataset.i; mark(); };
      });
    }
    const mark = () => $$(".ds-r", out).forEach((el) => { const on = +el.dataset.i === X.sel; el.classList.toggle("on", on); if (on) el.scrollIntoView({ block: "nearest" }); });
    async function run() {
      const text = q.value.trim(), my = ++X.seq;
      if (!text) { X.res = []; return draw(""); }
      X.res = match(local(), text).slice(0, 8); X.sel = 0; draw(text, true);
      const srcs = (S.opts.search || {}).sources || [];
      const more = (await Promise.all(srcs.map((f) => Promise.resolve().then(() => f(text)).catch(() => [])))).flat();
      if (my !== X.seq) return;
      X.res = [...X.res, ...more];
      draw(text);
    }
    function go(r) {
      if (!r) return;
      close();
      if (r.run) return r.run(r);
      if (r.href) location.href = r.href;
    }
    function open() { box.classList.add("open"); scrim.classList.add("on"); top.classList.add("ds-searching"); out.hidden = false; q.setAttribute("aria-expanded", "true"); if (!q.value) draw(""); }
    function close() { box.classList.remove("open"); scrim.classList.remove("on"); top.classList.remove("ds-searching"); out.hidden = true; q.setAttribute("aria-expanded", "false"); q.blur(); }
    scrim.onclick = close;
    q.addEventListener("focus", () => { open(); if (q.value) run(); });
    q.addEventListener("blur", () => setTimeout(() => { if (document.activeElement !== q) close(); }, 120));
    q.addEventListener("input", () => { clearTimeout(X.t); X.t = setTimeout(run, 110); });
    q.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); X.sel = Math.min(X.res.length - 1, X.sel + 1); mark(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); X.sel = Math.max(0, X.sel - 1); mark(); }
      else if (e.key === "Enter") { e.preventDefault(); go(X.res[X.sel]); }
      else if (e.key === "Escape") { e.preventDefault(); q.value = ""; close(); }
    });
    document.addEventListener("keydown", (e) => {
      const t = e.target, typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
      if ((!typing && !e.metaKey && !e.ctrlKey && !e.altKey && e.key === "/") || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k")) { e.preventDefault(); q.focus(); q.select(); }
    });
  }

  // ------------------------------------------------------------ assembly
  let rt = 0;
  function refresh() { clearTimeout(rt); rt = setTimeout(buildToc, 120); }

  function init(opts = {}) {
    S.opts = { routeTabs: true, markdown: true, toc: true, key: "ds", ...opts };
    const top = $(".ds-top");
    if (!top) { console.warn("docs-shell: no .ds-top header"); return; }
    let tl = $(".ds-tools", top);
    if (!tl) { tl = document.createElement("div"); tl.className = "ds-tools"; top.appendChild(tl); }
    theme(tl);
    search(top);
    const setH = () => root.style.setProperty("--ds-hdr", `${top.getBoundingClientRect().height}px`);
    setH();
    if (window.ResizeObserver) new ResizeObserver(setH).observe(top);

    const saved = parseInt(store.get(`${S.opts.key}-rail`), 10);
    for (const v of views()) {
      const rail = $(":scope > .ds-rail", v);
      if (rail) handle(v, rail); else v.classList.add("ds-no-rail");
      if (!$(":scope > .ds-toc", v)) { const toc = document.createElement("aside"); toc.className = "ds-toc"; toc.setAttribute("aria-label", "On this page"); v.appendChild(toc); }
      const doc = $(":scope > .ds-doc", v);
      if (doc) new MutationObserver(refresh).observe(doc, { childList: true, subtree: true, characterData: true });
    }
    setRail(Number.isFinite(saved) ? saved : RAIL.def, false);
    addEventListener("resize", () => { if (!root.classList.contains("ds-rail-off")) setRail(railNow(), false); });
    let raf = 0;
    addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; spy(); }); }, { passive: true });

    if (S.opts.routeTabs !== false) {
      $$(".ds-tabs [data-ds-tab]").forEach((t) => (t.onclick = (e) => { e.preventDefault(); show(t.dataset.dsTab); }));
      const want = (location.hash.match(/^#\/([\w-]+)/) || [])[1];
      const first = ($(".ds-tabs [data-ds-tab][aria-selected='true']") || $(".ds-tabs [data-ds-tab]") || {}).dataset;
      if (want || first) show(want || first.dsTab, false);
      addEventListener("hashchange", () => { const m = location.hash.match(/^#\/([\w-]+)/); if (m) show(m[1], false); });
    }
    buildToc();
  }

  window.DocsShell = { init, refresh, show, toast, markdown, setRail, jump };
})();
