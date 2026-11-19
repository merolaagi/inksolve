# InkSolve

Handwritten maths in; a reading that makes sense and a worked, checked answer out.

Draw on the page, upload a photo or scan, or type. InkSolve splits the ink into marks,
reads each mark with a small neural network, and then reads the whole line with
**context**: grammar, and (when solving) whether a sum is true. That is how a 7 written
like a 1 in `6 + 1 = 7` is read as 7, how `3x4` becomes 3 × 4 while `2x + 3` keeps its
letter x, and how a b and a 6 get told apart. Then SymPy solves it and shows the working.

```
./setup.sh      # once: a Python virtual environment with the packages
./run.sh        # http://127.0.0.1:48631 (or the next free port)
```

## Tabs

| Tab | What it does |
|---|---|
| **Solve** | Draw, photo or type. Each mark is shown with its top three guesses; correct one and InkSolve re-reads the line and saves the mark as a sample of your handwriting. **Solve** mode lets context fix marks; **Mark work** mode never uses whether the sum is true, and outlines unsure marks for a person to check. |
| **Your handwriting** | Write samples, guided towards the look-alikes (1/7, b/6, z/2, x/+). See what each symbol has been learnt from. Download MNIST and HASYv2. |
| **Train** | Train a reader from any mix of synthetic handwriting, MNIST, HASYv2 and yours. Compare models on writers they never saw: accuracy, 1→7 confusion, calibration. |
| **Experiment** | How few examples does a reader need? Trains with 5 to 1,000 real examples per symbol, from scratch, with augmentation, and from a synthetic-pretrained start, then measures single marks and whole sums with and without context, and how many mistakes each mode would hide when marking. |
| **Inside the model** | Every one of the reader's 140,696 weights as a block in 3D (height and colour = value). Draw a mark and watch the forward pass light up the weights that do the work, the backward pass give each weight its gradient, and the update move them. Click a block to follow one weight through the arithmetic. Train on batches live, from the trained reader or from random weights, and see a slider that scales the same picture to 20 billion parameters: memory, operations per token, training time, and where the weights sit in a transformer. |
| **Code** | Python Tutor for a neural network. The Python of one training step (`server/trace_code.py`: forward pass, loss, `backward()`, the last layer's gradient by hand, the update) really runs on the real weights under a line tracer. Step forward and back: the line that just ran and the next one are marked, the frames show every variable, arrows point to each tensor drawn as a heatmap, and the model's 24 weight and bias tensors show their gradients appearing and their values moving, layer by layer. |
| **How it works** | The method, in plain words. |

## What it reads

Digits, `+ − × ÷ / = ( )`, the letters `x y a b z`, decimal points, exponents (written
raised), fractions (written over a bar), and several lines as a system of equations.

It solves arithmetic (step by step), true/false statements, linear and quadratic
equations (factorising or the formula), other equations in one unknown, systems, and
rearranges one equation in two unknowns. Every answer is substituted back to check it.

## Data

Everything stays in `data/` on this computer: downloaded datasets, your samples (kept as
strokes), trained models, experiment results and the history.

* Synthetic handwriting: generated, every symbol, offline.
* [MNIST](http://yann.lecun.com/exdb/mnist/): 70,000 real digits (downloaded from a mirror).
* [HASYv2](https://zenodo.org/record/259444): 168,000 handwritten symbols; split by writer.
* Yours: the Your handwriting tab and corrections on Solve, each counted five times.

The starter model (synthetic + MNIST) reads real digits from unseen writers at 99.2%.
Operators and letters are learnt from synthetic handwriting until HASYv2 or your own
samples are added, so their accuracy on real writing is only known after that.

## Options

| Variable | Default | |
|---|---|---|
| `INKSOLVE_PORT` | 48631 | first port tried |
| `INKSOLVE_HOST` | 127.0.0.1 | |
| `INKSOLVE_DATA` | `./data` | |
| `INKSOLVE_PASSWORD` | none | asks for this password (any user name) if set |

Tests: `.venv/bin/python tests/test_core.py`.
