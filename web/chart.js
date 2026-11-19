/* A small SVG line chart with a hover tooltip, legend and direct end labels.
   lineChart(el, { x: [..], xLabel, yLabel, yFmt, logX, series: [{ name, color, values: [...], dash }] }) */
(function () {
  "use strict";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function lineChart(el, o) {
    const W = 640, H = 260, m = { l: 46, r: 120, t: 12, b: 38 };
    const xs = o.x, n = xs.length;
    const fx = o.logX ? (v) => Math.log10(v) : (v) => v;
    const x0 = fx(xs[0]), x1 = fx(xs[n - 1]) === x0 ? x0 + 1 : fx(xs[n - 1]);
    const all = o.series.flatMap((s) => s.values).filter((v) => v != null);
    let lo = o.yMin ?? Math.min(...all), hi = o.yMax ?? Math.max(...all);
    if (o.yMin == null) lo = Math.max(0, Math.floor((lo - 0.05) * 10) / 10);
    if (hi === lo) hi = lo + 1;
    const X = (v) => m.l + ((fx(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    const fmt = o.yFmt || ((v) => v.toFixed(2));
    const ticks = 5;
    let g = "";
    for (let i = 0; i <= ticks; i++) {
      const v = lo + ((hi - lo) * i) / ticks, y = Y(v);
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="var(--ds-rule)" stroke-width="1"/>`;
      g += `<text x="${m.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--ds-dim)">${esc(fmt(v))}</text>`;
    }
    xs.forEach((v) => { g += `<text x="${X(v)}" y="${H - m.b + 16}" text-anchor="middle" font-size="11" fill="var(--ds-dim)">${esc(v)}</text>`; });
    if (o.xLabel) g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" font-size="11.5" fill="var(--ds-dim)">${esc(o.xLabel)}</text>`;
    // direct end labels, nudged apart
    const ends = o.series.map((s) => { const i = s.values.map((v, j) => (v == null ? -1 : j)).filter((j) => j >= 0).pop(); return { s, y: Y(s.values[i]) }; })
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
    o.series.forEach((s) => {
      const pts = s.values.map((v, i) => (v == null ? null : [X(xs[i]), Y(v)])).filter(Boolean);
      g += `<polyline points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" ${s.dash ? 'stroke-dasharray="5 4"' : ""}/>`;
      pts.forEach((p) => { g += `<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${s.color}" stroke="var(--ds-bg)" stroke-width="2"/>`; });
    });
    ends.forEach((e) => { g += `<text x="${W - m.r + 10}" y="${e.y + 4}" font-size="11.5" fill="var(--ds-text)">${esc(e.s.name)}</text>`; });
    g += `<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--ds-faint)" stroke-width="1" visibility="hidden"/>`;
    g += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>`;
    el.classList.add("chart");
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "chart")}">${g}</svg><div class="tip" hidden></div>`;
    if (o.series.length > 1) {
      const lg = document.createElement("div");
      lg.className = "legend";
      lg.innerHTML = o.series.map((s) => `<span><i class="${s.dash ? "dash" : ""}" style="background:${s.color};color:${s.color}"></i>${esc(s.name)}</span>`).join("");
      el.prepend(lg);
    }
    const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), cross = el.querySelector(".cross");
    svg.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
      let best = 0;
      xs.forEach((v, i) => { if (Math.abs(X(v) - px) < Math.abs(X(xs[best]) - px)) best = i; });
      cross.setAttribute("x1", X(xs[best])); cross.setAttribute("x2", X(xs[best])); cross.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.innerHTML = `<b>${esc(o.xName || "")} ${esc(xs[best])}</b>` + o.series.map((s) => `<div><span style="color:${s.color}">●</span> ${esc(s.name)}: ${s.values[best] == null ? "–" : esc(fmt(s.values[best]))}</div>`).join("");
      const left = (X(xs[best]) / W) * r.width;
      tip.style.left = `${Math.min(left + 12, r.width - tip.offsetWidth)}px`;
      tip.style.top = `8px`;
    });
    svg.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  }
  window.lineChart = lineChart;
})();
