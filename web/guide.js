/* The Guide: an overlay that explains whatever is on screen, step by step, and can do
   things for you (jump, run a step, walk through on its own). Tabs report what is
   happening with Guide.context(tab, facts); the Guide turns facts into explanations.
   Questions are answered from a built-in glossary, or by Claude when an Anthropic API key
   has been saved on this computer. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => (typeof n === "number" ? Math.round(n).toLocaleString() : n);
  const store = { get(k) { try { return localStorage.getItem("inksolve-guide-" + k); } catch { return null; } }, set(k, v) { try { localStorage.setItem("inksolve-guide-" + k, v); } catch { /* private */ } } };
  const G = { tab: null, facts: null, depth: store.get("depth") || "simple", speak: store.get("speak") === "1", walking: false, ai: false, open: store.get("open") ? store.get("open") === "1" : innerWidth >= 640 };

  // ================================================================== what each step means
  // Each entry: title, simple, detail, why (optional), look [[label, selector]], scale (optional), next action
  const CODE = {
    start: (f) => ({ title: "One training step, about to run",
      simple: `We'll push one handwritten mark through the network, measure how wrong it is, work out how every weight should change, then change them all a little. That is all training is, repeated millions of times.`,
      detail: `<code>training_step()</code> gets four things: the model (<code>net</code>), the image (28 × 28 numbers, how much ink in each pixel), the answer we claim is right (“${esc(f.label)}”), and the learning rate (${f.lr}), which sets how big each change is.`,
      why: "Every neural network, including 20-billion-parameter language models, learns with exactly this loop.",
      look: [["The variables", "#cd-frames"], ["The model's weights", "#cd-model"], ["The code", "#cd-lines"]] }),
    zero_grad: () => ({ title: "Clear last step's gradients",
      simple: "Every weight has a slot, <code>.grad</code>, that will hold how it should change. First we empty them.",
      detail: "PyTorch adds new gradients onto whatever is already there, so without this the last step would leak into this one. All 24 tensors (a weight and a bias for each of the 12 layers) now show None in the .grad column.",
      look: [["The .grad column", "#cd-model"]] }),
    after_forward: () => ({ title: "forward() is done", simple: "The image has been through every layer. <code>logits</code> holds 22 scores, one per symbol, and <code>a</code> the 128 neuron values we'll need later.", look: [["logits", '#cd-objects [data-obj]']] }),
    conv: (f) => ({ title: `${f.layer}: ${f.filters} filters slide over the image`,
      simple: "Each filter is a small 3 × 3 stencil of weights. It slides to every position, multiplies what's under it by its weights and adds it up. Where the ink matches the stencil, the result is big.",
      detail: `<code>${f.layer}.weight</code> holds ${f.filters} filters, each looking at ${f.inC} input map${f.inC > 1 ? "s" : ""} through a 3 × 3 window: ${f.per} weights per filter, ${fmt(f.weights)} in all. Sliding over ${fmt(f.positions)} positions gives <code>h</code> = ${f.shape} (${fmt(f.n)} numbers) and costs ${fmt(f.macs)} multiply-adds. The same few weights are reused everywhere: cheap in weights, expensive in arithmetic.`,
      why: "Early filters respond to edges and strokes; later ones combine those into curves, corners and parts of symbols.",
      look: [["h, the new maps", "#cd-objects .t-obj"], [`${f.layer} in the model`, `#cd-model [data-layer="${f.layer}"]`]],
      scale: "A language model has no convolutions, but the same idea: a weight matrix applied to every token. A 20B model does about 40 billion multiply-adds per token." }),
    bnrelu: (f) => ({ title: "Rescale, then switch off the negatives",
      simple: "BatchNorm rescales each map so the numbers stay in a sensible range. ReLU then replaces every negative number with 0.",
      detail: `${f.zeros}% of <code>h</code> is now exactly 0. Those positions pass nothing forward, and on the way back they pass no gradient either, so weights that only fed them won't change this step.`,
      why: "Without a bend like ReLU, any stack of layers would collapse into one straight-line rule. The bends are what let a network learn shapes.",
      look: [["h (look for the flat zero areas)", "#cd-objects .t-obj"]] }),
    pool: (f) => ({ title: "Max pooling: half the size", simple: "Each 2 × 2 square keeps only its largest number.",
      detail: `${esc(f.from)} → ${esc(f.to)}. Pooling has no weights. It makes the network care that a feature is roughly here rather than exactly here, and lets the next filters see a wider area.`, look: [["h", "#cd-objects .t-obj"]] }),
    flatten: (f) => ({ title: "Flatten", simple: `The 64 small maps are laid out as one row of ${fmt(f.n)} numbers, ready for ordinary matrix multiplication.`, detail: "Nothing is computed: the same numbers, a different shape. From here on the network forgets where things were in the picture." }),
    fc1: (f) => ({ title: "fc1: a 128 × 576 matrix multiplication",
      simple: "Each of 128 neurons takes all 576 numbers, multiplies each by its own weight and adds them up: one row of the weight matrix times the input.",
      detail: `73,728 weights, more than half the model, each used exactly once here. ReLU then switches off ${f.off} of the 128 neurons for this mark; their whole rows of 576 weights get no gradient this step.`,
      look: [["a, the 128 neurons", "#cd-objects .t-obj"], ["fc1 in the model", '#cd-model [data-layer="fc1"]']],
      scale: "This is what most of a 20B language model is: big matrices like this (about 6,000 × 24,000 in each feed-forward block), applied to every token." }),
    fc2: (f) => ({ title: "fc2: one score per symbol", simple: "22 rows of 128 weights. Row k times the neurons gives the score for symbol k.",
      detail: `These scores (logits) aren't probabilities yet: any size, positive or negative. The biggest is the model's guess: “${esc(f.guess)}”.`, look: [["fc2 in the model", '#cd-model [data-layer="fc2"]']] }),
    return_forward: () => ({ title: "Returning from forward()", simple: "forward() hands back the scores and the neuron values. Its frame closes, and its local <code>h</code> and <code>v</code> are gone.", look: [["Frames", "#cd-frames"]] }),
    softmax: (f) => ({ title: "From scores to probabilities", simple: "Softmax makes every score positive and scales them so they add up to 1.",
      detail: `Top three: ${f.top}. The answer we're teaching, “${esc(f.label)}”, has ${f.p}.`, why: f.guess === f.label ? "The model already agrees, so this step will be gentle." : `The model thinks “${esc(f.guess)}”, so this step will push hard.`, look: [["probs", "#cd-objects .t-obj"]] }),
    loss: (f) => ({ title: "How wrong was it?", simple: `<code>loss</code> = −log(probability given to “${esc(f.label)}”) = <b>${f.loss}</b>.`,
      detail: "If the right answer had probability 1, the loss would be 0. At 50% it's 0.69, at 1% it's 4.6. Everything that follows exists to make this number smaller.",
      why: "This is cross-entropy, the same objective used to train language models, where the right answer is the next word.", look: [["loss", "#cd-objects .t-obj"]] }),
    backward: (f) => ({ title: "Backpropagation",
      simple: "One line, a lot of work: PyTorch walks back through everything forward() did and, with the chain rule, works out for every weight: if I nudged this up a little, would the loss go up or down, and how much?",
      detail: `The answers are the gradients, stored in .grad with the same shape as each weight; ${f.n} tensors now have one. Average size by layer: ${esc(f.per)}. ${fmt(f.zero)} weights got exactly zero: they fed only neurons that ReLU had switched off.`,
      why: "Backward costs about twice the forward pass. That's why training a model costs about three times as much per token as running it.",
      look: [["The .grad column, now filled", "#cd-model"]], scale: "In a 20B model this line produces 20 billion gradients, one per weight, every step." }),
    error: (f) => ({ title: "The output error", simple: `What the model said minus what it should have said: 1 for “${esc(f.label)}”, 0 for every other symbol.`,
      detail: `For softmax with cross-entropy, this simple difference is exactly the gradient of the loss with respect to the 22 scores. Negative at “${esc(f.label)}” (push its score up), positive elsewhere (push those down), largest where the model was most confidently wrong.`, look: [["error", "#cd-objects .t-obj"]] }),
    grad_fc2: (f) => ({ title: "Checking backward() by hand",
      simple: "fc2's gradient is just error × neuron value, for every pair: 22 × 128 numbers.",
      detail: `Weight [k, j] connects neuron j to symbol k. Its gradient is error[k] × a[j]: big when the neuron was active and that symbol's score was wrong. Where neuron j was off, the whole column is zero (${f.zeros}% of this matrix): no part in the answer, no blame. Compare grad_fc2 with fc2's .grad below: they are the same numbers.`,
      look: [["grad_fc2", "#cd-objects .t-obj"], ["fc2's .grad", '#cd-model [data-layer="fc2"]']] }),
    no_grad: () => ({ title: "Stop recording", simple: "Inside this block PyTorch stops tracking operations: the update itself is not something the network should learn through." }),
    length: (f) => ({ title: "How big is the whole gradient?", simple: `All ${fmt(f.total)} gradient numbers, treated as one long arrow. Its length is ${f.len}.` }),
    scale: (f) => ({ title: "Gradient clipping", simple: f.scale < 1 ? `The arrow is longer than 1, so every gradient is shortened by the same factor (${f.scale.toPrecision(3)}). One odd example can't throw the model far.` : "Short enough: the gradients are used as they are." }),
    for: (f) => ({ title: f.layer ? `Next layer: ${f.layer}` : "Every layer updated", simple: f.layer ? `The loop moves on to <code>net.${esc(f.layer)}</code>.` : "The loop is finished.", look: f.layer ? [[f.layer, `#cd-model [data-layer="${f.layer}"]`]] : [] }),
    update: (f) => ({ title: `${f.layer}.${f.which} moves`,
      simple: "Every number takes a small step downhill: new = old − learning rate × gradient.",
      detail: `${fmt(f.n)} numbers, ${fmt(f.changed)} actually changed (the rest had zero gradient); the biggest move was ±${f.maxmove}. With learning rate ${f.lr} and clip factor ${Number(f.scale).toPrecision(3)}, a weight with gradient 1 moves by ${(f.lr * f.scale).toPrecision(3)}.`,
      why: "Steps are tiny on purpose: the model sees many examples, and each should only nudge it.",
      look: [[`${f.layer}: new value and the moves`, `#cd-model [data-layer="${f.layer}"]`]],
      scale: "In a 20B model this updates 20 billion numbers each step. Adam also keeps two running averages per weight, which is why training needs about 16 bytes per parameter." }),
    done: (f) => ({ title: "Step complete", simple: `All ${fmt(f.total)} numbers moved a little. Show the same mark again and the model is a bit more likely to say “${esc(f.label)}”.`,
      detail: `The loss was ${f.loss}. Training is this, thousands to millions of times, over many different examples.` }),
    other: (f) => ({ title: "Running", simple: f.code ? `<code>${esc(f.code)}</code>` : "" }),
  };
  const INSIDE = {
    idle: (f) => ({ title: "Every weight, as a block", simple: `Each of the ${fmt(f.shown || 140696)} small bars is one weight. Height and colour are its value: orange positive, blue negative (hanging below the floor). Between them are what each layer produced for your mark.`, detail: "Drag to turn the view, scroll to zoom, click a block to follow one weight. The panel on the left steps through forward, backward and update." }),
    forward: (f) => ({ title: "Forward pass", simple: "Each layer multiplies what comes in by its weights, adds it up and passes it on. Bright weights are the ones doing the work for this mark.", detail: f.guess ? `It reads the mark as “${esc(f.guess)}” (${f.p}). About ${fmt(f.macs)} multiply-adds.` : "" }),
    backward: (f) => ({ title: "Backward pass", simple: "Starting from how wrong the answer was, the chain rule runs back through every layer and gives each weight its gradient: which way to move, and how hard.", detail: f.loss ? `Loss ${f.loss}. Grey weights got no gradient: they weren't used for this mark.` : "" }),
    update: (f) => ({ title: "Update", simple: "Every weight with a gradient moves a little. The heights are shown enlarged so you can see it, then settle to their real size.", detail: f.text || "" }),
    train: (f) => ({ title: "Training on batches", simple: "Each step now uses 32 marks at once and averages their gradients. Watch the loss fall and the weights shift.", detail: f.text || "" }),
    weight: (f) => ({ title: `Following one weight in ${esc(f.layer)}`, simple: "Scroll down to “One weight, followed” for its value, what it multiplied, its gradient and the arithmetic of its update." }),
  };
  const TABS = {
    solve: { title: "Solve", simple: "Write a sum or an equation on the pad, upload a photo, or type it. InkSolve reads each mark, uses the maths to settle close calls (a 1 that must be a 7), then solves it with the working shown.", detail: "Mark work mode reads exactly what was written, true or not, which is what you want when checking someone's answers." },
    hand: { title: "Your handwriting", simple: "A few dozen of your own 1s and 7s teach the reader your hand better than thousands of strangers'. Write what it asks for; each sample counts five times when training." },
    train: { title: "Train", simple: "Train a new reader from any mix of synthetic handwriting, MNIST, HASYv2 and your samples, and compare models on writers they never saw." },
    experiment: { title: "Experiment", simple: "How few examples does a reader need? This trains many readers with less and less data, with and without the tricks that help, and measures them on unseen writers." },
    code: { title: "Code, step by step", simple: "The Python of one real training step, run line by line on the real weights. Step with ← →, or press ▶ Walk me through and I'll explain each line as it runs." },
    inside: { title: "Inside the model", simple: "Every weight of the reader as a block in 3D. Press One full training step to watch forward, backward and update play over them." },
    how: { title: "How it works", simple: "The method in plain words: marks, the symbol model, context, marking work, fewer examples, solving." },
  };
  const ACTIONS = {
    code: [["▶ Walk me through", () => walk()], ["Jump to backward", () => Tutor.phase("Backward pass")], ["Jump to update", () => Tutor.phase("Update")], ["New step: say a 7 is a 1", () => Tutor.run("1", "7")]],
    inside: [["One full training step", () => $("#in-all")?.click()], ["Show me “use”", () => $('[data-cmode="use"]')?.click()], ["Learn from random", () => { $("#in-reset-r")?.click(); setTimeout(() => $("#in-play")?.click(), 1500); }], ["See the code", () => DocsShell.show("code")]],
    solve: [["Try the 1-or-7 example", () => $('[data-ex="ambiguous-7"]')?.click()], ["Mark work mode", () => $('[data-mode="grade"]')?.click()], ["Look inside the model", () => DocsShell.show("inside")]],
    hand: [["Write samples", () => DocsShell.jump?.("write-samples")]],
    train: [["See the code of one step", () => DocsShell.show("code")]],
    experiment: [["See it in 3D", () => DocsShell.show("inside")]],
    how: [["See the code", () => DocsShell.show("code")], ["See it in 3D", () => DocsShell.show("inside")]],
  };

  // ================================================================== offline answers
  const GLOSSARY = [
    [["gradient", "grad"], "A gradient says, for one weight, how the loss would change if that weight were nudged up a little. Positive: raising it makes things worse, so the update lowers it. Negative: the opposite. Zero: this weight played no part in this answer."],
    [["loss", "cross entropy", "cross-entropy"], "The loss is one number saying how wrong the model was: −log of the probability it gave the right answer. 0 means certain and right; it grows quickly as the right answer gets less likely."],
    [["backprop", "backward", "chain rule"], "Backpropagation works out every gradient in one sweep from the output back to the input. Each layer passes back ‘how much the loss changes with my output’, and multiplies it by how its output changes with its weights and inputs: that's the chain rule."],
    [["learning rate", "lr", "step size"], "The learning rate scales every update: new = old − learning rate × gradient. Too small and learning crawls; too big and the weights overshoot and the loss jumps around or blows up."],
    [["relu"], "ReLU keeps positive numbers and turns negative ones into 0. It's the bend that lets stacked layers learn more than a straight-line rule, and it's also why many weights get zero gradient: an output of 0 passes no gradient back."],
    [["batchnorm", "batch norm", "normalize", "normalise"], "BatchNorm rescales each feature map to a steady range using statistics learnt during training, then applies a learnt scale and shift (its weight and bias). It makes training faster and more stable."],
    [["convolution", "conv", "filter", "kernel"], "A convolution slides a small grid of weights (a 3 × 3 kernel per input map) over every position and adds up weight × pixel. The same weights are reused everywhere, so a filter that detects an edge finds it anywhere in the picture."],
    [["pool", "pooling", "max pool"], "Max pooling keeps the largest number in each 2 × 2 square, halving the size. It has no weights; it makes features less sensitive to exact position."],
    [["softmax", "probabilit"], "Softmax turns scores into probabilities: it raises e to each score and divides by the total, so they're all positive and add up to 1. Bigger score, bigger share."],
    [["logit", "score"], "Logits are the raw scores the last layer produces, one per symbol, before softmax. Only their differences matter for the probabilities."],
    [["weight", "parameter"], "A weight is one learnt number. Each layer multiplies its input by its weights and adds them up. This model has 141,470 parameters: 140,696 weights plus biases and BatchNorm numbers. A 20B model has 20 billion."],
    [["bias"], "A bias is a learnt number added after the weights are summed, one per output. It lets a neuron switch on even when its input is small, or stay off unless the input is large."],
    [["tensor", "shape"], "A tensor is a grid of numbers with any number of dimensions. 1×24×28×28 means one image, 24 maps, each 28 × 28. Weights are tensors too: conv2's is 24×24×3×3."],
    [["sgd", "stochastic"], "SGD (stochastic gradient descent) is the plain update: each weight moves by −learning rate × its gradient, computed on one example or a small batch."],
    [["adam"], "Adam is SGD with memory: for each weight it keeps a running average of recent gradients and of their squares, and scales each step by those. It often learns faster, at the cost of two extra numbers per weight."],
    [["clip", "clipping"], "Gradient clipping: if all gradients together are longer than a limit (1 here), they're all shrunk by the same factor. It stops one strange example from making a huge jump."],
    [["one hot", "one-hot"], "One-hot is a list of 22 zeros with a single 1 at the right answer. Subtracting it from the probabilities gives the error at the output."],
    [["batch", "epoch"], "A batch is a group of examples whose gradients are averaged for one update (32 in the Inside tab). An epoch is one pass through the whole training set."],
    [["20b", "20 billion", "billion", "large model", "llm", "transformer"], "A 20-billion-parameter language model trains with exactly this loop. The differences are scale (20 billion weights instead of 141 thousand), shape (mostly attention and feed-forward matrices instead of convolutions) and the target (the next word instead of a symbol). The Inside the model tab's slider works out the memory and compute."],
    [["overfit"], "Overfitting is when a model learns its training examples so well, including their quirks, that it does worse on new ones. Augmentation and more varied data help."],
    [["fc1", "fc2", "fully connected", "linear", "matrix"], "A fully connected layer is one matrix multiplication: each output is one row of weights times the whole input vector, plus a bias. fc1 is 128 × 576, fc2 is 22 × 128."],
    [["zero_grad", "zero grad"], "zero_grad() empties every .grad before a new step, because PyTorch adds gradients onto whatever is already there."],
    [["no_grad", "no grad"], "torch.no_grad() tells PyTorch not to record operations, so the update itself doesn't become part of what gets differentiated."],
    [["why", "zero gradient", "didn't move", "not move"], "A weight gets zero gradient when it had no effect on this answer: usually because the neuron it feeds was switched off by ReLU, or the input it multiplies was 0. No effect, no blame, no change."],
  ];
  function offline(q) {
    const s = q.toLowerCase();
    const hits = GLOSSARY.map(([keys, a]) => [keys.reduce((n, k) => n + (s.includes(k) ? k.length : 0), 0), a]).filter(([n]) => n).sort((a, b) => b[0] - a[0]);
    return hits.length ? hits.filter((h, i) => i === 0 || h[0] >= hits[0][0]).slice(0, 2).map((h) => h[1]).join("\n\n") : null;
  }

  // ================================================================== the window
  function build() {
    const el = document.createElement("div");
    el.className = "guide";
    el.id = "guide";
    el.setAttribute("role", "complementary");
    el.setAttribute("aria-label", "Guide");
    el.innerHTML = `
      <div class="g-head" id="g-head"><span class="g-dot"></span><b>Guide</b><span class="g-ctx" id="g-ctx"></span>
        <span class="g-tools"><button id="g-depth" title="Simple or detailed"></button><button id="g-speak" title="Read aloud"></button><button id="g-min" title="Minimise" aria-label="Minimise">–</button></span></div>
      <div class="g-body" id="g-body"></div>
      <div class="g-acts" id="g-acts"></div>
      <form class="g-ask" id="g-ask"><input id="g-q" placeholder="Ask about this step…" autocomplete="off"><button class="ds-btn small" type="submit">Ask</button><button type="button" class="g-gear" id="g-gear" title="Claude settings">⚙</button></form>
      <div class="g-ans" id="g-ans" hidden></div>`;
    document.body.appendChild(el);
    const fab = document.createElement("button");
    fab.className = "guide-fab";
    fab.id = "guide-fab";
    fab.innerHTML = '<span class="g-dot"></span> Guide';
    fab.title = "Open the Guide";
    document.body.appendChild(fab);
    fab.onclick = () => show(true);
    $("#g-min").onclick = () => show(false);
    $("#g-depth").onclick = () => { G.depth = G.depth === "simple" ? "detail" : "simple"; store.set("depth", G.depth); render(); };
    $("#g-speak").onclick = () => { G.speak = !G.speak; store.set("speak", G.speak ? "1" : "0"); if (!G.speak) speechSynthesis?.cancel(); else say(); paintTools(); };
    $("#g-ask").onsubmit = (e) => { e.preventDefault(); ask($("#g-q").value); };
    $("#g-gear").onclick = settings;
    drag(el, $("#g-head"));
    const pos = store.get("pos");
    if (pos) { try { const [x, y] = JSON.parse(pos); place(el, x, y); } catch { /* ignore */ } }
    show(G.open);
    paintTools();
  }
  function paintTools() {
    $("#g-depth").textContent = G.depth === "simple" ? "Simple" : "Detailed";
    $("#g-speak").textContent = G.speak ? "🔊" : "🔈";
    $("#g-speak").title = G.speak ? "Stop reading aloud" : "Read aloud";
    if (!("speechSynthesis" in window)) $("#g-speak").hidden = true;
  }
  function show(open) {
    G.open = open;
    store.set("open", open ? "1" : "0");
    $("#guide").hidden = !open;
    $("#guide-fab").hidden = open;
    if (!open) { stopWalk(); speechSynthesis?.cancel?.(); }
  }
  function place(el, x, y) {
    const w = el.offsetWidth || 380, h = el.offsetHeight || 300;
    x = Math.max(4, Math.min(innerWidth - w - 4, x)); y = Math.max(60, Math.min(innerHeight - 60, y));
    el.style.left = `${x}px`; el.style.top = `${y}px`; el.style.right = "auto"; el.style.bottom = "auto";
  }
  function drag(el, handle) {
    handle.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button") || innerWidth < 640) return;
      const r = el.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top;
      handle.setPointerCapture(e.pointerId);
      const mv = (ev) => place(el, ev.clientX - dx, ev.clientY - dy);
      const up = (ev) => { handle.removeEventListener("pointermove", mv); handle.removeEventListener("pointerup", up); store.set("pos", JSON.stringify([ev.clientX - dx, ev.clientY - dy])); };
      handle.addEventListener("pointermove", mv);
      handle.addEventListener("pointerup", up);
    });
    handle.addEventListener("dblclick", () => { el.style.left = el.style.top = ""; el.style.right = ""; el.style.bottom = ""; store.set("pos", ""); });
  }

  // ================================================================== rendering
  function current() {
    const f = G.facts || {};
    if (G.tab === "code" && f.kind) return (CODE[f.kind] || CODE.other)(f);
    if (G.tab === "inside" && f.kind) return (INSIDE[f.kind] || INSIDE.idle)(f);
    if (G.tab === "solve" && f.kind === "reading") return readingText(f);
    return TABS[G.tab] || { title: "InkSolve", simple: "" };
  }
  function readingText(f) {
    const out = { title: `Read: ${esc(f.text)}`, simple: f.changed?.length ? "Context changed a mark: " + f.changed.map(esc).join(" ") : "Every mark was read on its shape alone; context agreed.",
      detail: `${f.marks} marks, confidence ${f.conf}. ${f.truth != null ? (f.truth ? "The statement is true as written." : "The statement is false as written.") : ""} ${f.mode === "grade" ? "Mark work mode: truth was not used to read it." : "Solve mode: a false sum is taken to be less likely than a true one, which is what lets a 1-like mark become a 7."}`,
      look: [["The marks", "#marks"], ["The reading", "#reading"]] };
    return out;
  }
  function render(speak = true) {
    if (!$("#guide")) return;
    const c = current(), f = G.facts || {};
    $("#g-ctx").textContent = G.tab === "code" && f.step ? `Code · step ${f.step} of ${f.of}` : G.tab ? ({ code: "Code", inside: "Inside the model", solve: "Solve", hand: "Your handwriting", train: "Train", experiment: "Experiment", how: "How it works" }[G.tab] || "") : "";
    let h = `<h4>${c.title || ""}</h4>`;
    if (c.simple) h += `<p>${c.simple}</p>`;
    if (G.depth === "detail") {
      if (c.detail) h += `<p>${c.detail}</p>`;
      if (c.why) h += `<p class="g-why"><b>Why it matters.</b> ${c.why}</p>`;
      if (c.scale) h += `<p class="g-scale"><b>At 20 billion.</b> ${c.scale}</p>`;
    } else if (c.detail || c.why) h += `<p class="g-more"><button id="g-moreb">More detail ›</button></p>`;
    if (c.look?.length) h += `<div class="g-look">Look at: ${c.look.map(([l, sel], i) => `<button data-look="${i}">${esc(l)}</button>`).join("")}</div>`;
    $("#g-body").innerHTML = h;
    $("#g-moreb")?.addEventListener("click", () => { G.depth = "detail"; store.set("depth", "detail"); paintTools(); render(); });
    $$("#g-body [data-look]").forEach((b) => {
      const sel = c.look[+b.dataset.look][1];
      b.onmouseenter = () => hl(sel, true); b.onmouseleave = () => hl(sel, false);
      b.onclick = () => { const t = $(sel); if (t) { t.scrollIntoView({ behavior: "smooth", block: "center" }); hl(sel, true); setTimeout(() => hl(sel, false), 1600); } };
    });
    const acts = ACTIONS[G.tab] || [];
    $("#g-acts").innerHTML = acts.map(([l], i) => `<button data-act="${i}">${esc(G.walking && i === 0 && G.tab === "code" ? "❚❚ Stop walking" : l)}</button>`).join("");
    $$("#g-acts [data-act]").forEach((b) => (b.onclick = () => { const a = acts[+b.dataset.act]; if (G.walking && +b.dataset.act === 0 && G.tab === "code") stopWalk(); else a[1](); }));
    paintTools();
    if (speak) say();
  }
  function hl(sel, on) { $$(sel).slice(0, 3).forEach((e) => e.classList.toggle("guide-hl", on)); }
  function plain(html) { const d = document.createElement("div"); d.innerHTML = html; return d.textContent; }
  function say() {
    if (!G.speak || !("speechSynthesis" in window) || !G.open) return;
    const c = current();
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(plain(`${c.title || ""}. ${c.simple || ""} ${G.depth === "detail" ? c.detail || "" : ""}`));
    u.rate = 1.04;
    u.onend = () => { if (G.walking && G.utter === u) setTimeout(advance, 500); };
    G.utter = u;
    speechSynthesis.speak(u);
  }

  // ================================================================== walking through on its own
  function walk() {
    if (!window.Tutor || !Tutor.at()) return;
    G.walking = true;
    if (Tutor.at().k >= Tutor.at().n - 1) Tutor.go(0);
    render();
    if (!G.speak) G.timer = setTimeout(advance, delayFor());
  }
  function delayFor() { const c = current(); return Math.min(9000, 2200 + plain(`${c.simple || ""} ${G.depth === "detail" ? c.detail || "" : ""}`).length * 24); }
  function advance() {
    if (!G.walking) return;
    if (!Tutor.next()) { stopWalk(); return; }
    if (!G.speak) G.timer = setTimeout(advance, delayFor());
  }
  function stopWalk() { const was = G.walking; G.walking = false; clearTimeout(G.timer); if (was && $("#g-acts")) render(false); }
  addEventListener("keydown", (e) => { if (G.walking && (e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "Escape")) stopWalk(); });

  // ================================================================== asking
  async function ask(q) {
    q = (q || "").trim();
    if (!q) return;
    const box = $("#g-ans");
    box.hidden = false;
    box.innerHTML = `<p class="g-q">${esc(q)}</p><p class="g-wait">Thinking…</p>`;
    const local = offline(q);
    if (G.ai) {
      try {
        const r = await fetch("/api/guide/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q, tab: G.tab, facts: G.facts, explanation: plain(Object.values(current()).filter((v) => typeof v === "string").join(" ")) }) });
        const j = await r.json();
        if (!r.ok) throw new Error(j.detail || "failed");
        box.innerHTML = `<p class="g-q">${esc(q)}</p>${j.answer.split(/\n{2,}/).map((p) => `<p>${esc(p)}</p>`).join("")}<p class="g-src">Answered by Claude, using this step's numbers.</p>`;
        return;
      } catch (e) {
        if (!local) { box.innerHTML = `<p class="g-q">${esc(q)}</p><p>Claude couldn't answer: ${esc(e.message)}</p>`; return; }
      }
    }
    box.innerHTML = `<p class="g-q">${esc(q)}</p>` + (local ? local.split("\n\n").map((p) => `<p>${esc(p)}</p>`).join("") + `<p class="g-src">From the built-in glossary.</p>` : `<p>That one isn't in the built-in glossary. Try words like gradient, loss, ReLU, learning rate, softmax, convolution or 20B. To ask anything, save an Anthropic API key with ⚙ (it stays on this computer).</p>`);
  }
  async function settings() {
    const box = $("#g-ans");
    box.hidden = false;
    box.innerHTML = `<p><b>Answers from Claude</b></p><p class="g-src">With an Anthropic API key saved on this computer, the Guide sends your question and the current step's numbers to Claude. The key is stored in data/settings.json and can only be set from this computer.</p>
      <div class="g-keyrow"><input id="g-key" type="password" placeholder="${G.ai ? "A key is saved. Paste a new one to replace it" : "sk-ant-…"}" autocomplete="off"><button class="ds-btn small" id="g-keysave">Save</button>${G.ai ? '<button class="ds-btn small" id="g-keydel">Remove</button>' : ""}</div>`;
    const send = async (key) => {
      try {
        const r = await fetch("/api/guide/key", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) });
        const j = await r.json();
        if (!r.ok) throw new Error(j.detail || "failed");
        G.ai = j.ai; box.innerHTML = `<p>${G.ai ? "Saved. Questions now go to Claude." : "Removed. Questions use the built-in glossary."}</p>`;
      } catch (e) { box.innerHTML = `<p>${esc(e.message)}</p>`; }
    };
    $("#g-keysave").onclick = () => send($("#g-key").value.trim());
    $("#g-keydel")?.addEventListener("click", () => send(""));
  }

  // ================================================================== wiring
  window.Guide = {
    context(tab, facts) {
      G.tab = tab;
      G.facts = facts;
      if (G.walking && tab !== "code") stopWalk();
      render();
    },
  };
  document.addEventListener("ds:view", (e) => { if (G.tab !== e.detail) { G.tab = e.detail; G.facts = null; stopWalk(); const a = $("#g-ans"); if (a) { a.hidden = true; a.innerHTML = ""; } render(); } });
  function start() {
    build();
    G.tab = (location.hash.replace("#/", "") || "solve");
    render();
    fetch("/api/health").then((r) => r.json()).then((h) => { G.ai = !!h.guide_ai; }).catch(() => {});
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
