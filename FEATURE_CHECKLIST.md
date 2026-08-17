# Feature Checklist

Checklist for adding a new processing feature (a QC check, a manual cut, a
correction, etc.) to `pyadps`. Split into the **Pythonic** work (the
importable library — works headless, no Streamlit) and the **Streamlit**
work (wiring it into the web app). Do Pythonic first; Streamlit consumes it.

Most items here exist because skipping them caused a real bug at some
point — see the inline notes.

## Pythonic Additions

Core library code lives in `src/pyadps/processing/*.py` (and `io/*.py` for
file-format concerns). Each processing stage exposes a `*Runner` class
(`VelocityCheckRunner`, `SignalQualityRunner`, `SensorHealthRunner`, ...)
built on a shared `_record_check()` helper, plus an `apply_pipeline()`
dict-driven entry point.

- [ ] **Core function.** Pure function operating on the dataset/arrays
  directly (e.g. `trim_depths()` in `velocity_check.py`). Validate inputs
  at the boundary (see `_validate_threshold` / `THRESHOLD_RANGES` pattern)
  rather than trusting the caller.
- [ ] **Default constants**, not inline literals — a single
  `DEFAULT_X = ...` near the top of the module (see
  `DEFAULT_DESPIKE_KERNEL` in `velocity_check.py`). Every other default
  below should point back at this one constant, not repeat the literal.
- [ ] **Runner method.** Add it to the relevant `*Runner` class via
  `_record_check(...)`, returning `self` for chaining. Note where it must
  sit in the chain (e.g. magnetic correction must run before anything
  reading velocity values) and say so in the docstring.
- [ ] **`apply_pipeline()`** — wire the new check into the dict-driven
  config path too, not just the fluent chain.
- [ ] **`ProcessingConfig`** (`config.py`): new dataclass field(s) with a
  default that matches the Runner method's default *exactly*. Check three
  places for drift, not one — this repo had a real kernel=5-vs-13 mismatch
  between the dataclass default and the Runner default that went unnoticed
  for a while:
  - dataclass field default
  - `from_ini()`'s `fallback=...`
  - `to_ini()` / `_build_configparser()` write-out
- [ ] **`ProcessedDataset.apply_*()`** (`core.py`): new params with the
  same default, docstring entry, and — if order matters — a comment
  stating the enforced order (e.g. "magnetic correction, depth trim,
  threshold, despike, flatline").
- [ ] **`apply_config()`** wiring so the INI-driven path reaches the new
  check.
- [ ] **`autoprocess()`** — usually needs nothing extra since it just calls
  `apply_config()`, but confirm rather than assume.
- [ ] **Docstrings** — NumPy style, defaults stated as the actual default,
  not stale copy-paste from a similar check.
- [ ] **Tests**, at every layer, not just the happy path:
  - the core function directly
  - the Runner method (including a `test_default_parameters` that asserts
    against the module constant, not a hardcoded literal)
  - `ProcessingConfig` roundtrip (`from_ini` → `to_ini` → `from_ini`)
  - `core.py` integration (`apply_velocity_check()`-style method)
- [ ] **Docs** (`docs/source/processing/*.md`): signature table, defaults,
  and the "checks apply in this order" note if it's order-sensitive.
- [ ] **`CHANGELOG.md`** `[Unreleased]` entry.

## Streamlit Additions

Pages live in `src/pyadps/pages/NN_Name.py`. A page is a single script that
Streamlit re-executes top-to-bottom on *every* interaction — all
`with tabN:` blocks run on every rerun regardless of which tab is visually
active, so state written in one tab is visible to a later tab in the same
run only if that tab's code runs *after* it in file order (tabs can be
visually reordered via `with tabN:` labels without moving the code).

- [ ] **UI element** (new tab, or a section inside one), matching the
  existing widget style: `st.session_state.x = st.checkbox(..., value=
  st.session_state.x, key="x_cb")`. Remember: once `key` exists in
  session_state, `value=`/`index=` are only honored on the *first* render
  ever — they will not force a widget back to a given value on a later
  rerun.
- [ ] **Session-state init block** (the page's `if not
  st.session_state.<page>_initialized:` block) — add every new key here.
- [ ] **Reset callback** (e.g. `_reset_velocity_tests()`) — reset the new
  keys here too, including the raw widget-backed `_cb` keys, not just the
  logical flag. Resetting only the logical flag and leaving the widget's
  own key untouched means the checkbox won't visually uncheck on the next
  render (the `value=` param is ignored once the key exists).
- [ ] **Guard/precondition branches** (e.g. "requires a regridded
  dataset") must reset *all* of the new feature's state keys when the
  precondition fails, the same way the reset callback does — a guard that
  only resets one of several related keys leaves the others stale.
- [ ] **Preview / staging button handler** — wire the new check in, in the
  position matching the documented pipeline order. This drifted from the
  documented order here before (Preview handler had depth trim last;
  docs/`apply_velocity_check()` say it runs right after magnetic
  correction) — diff the button handler's call order against the docs
  order explicitly, don't assume it matches.
- [ ] **Apply/Save button handler** — same wiring, same order.
- [ ] **Every settings-summary table** that mirrors this state — there is
  often more than one (this page has three: the Preview tab table, the
  Save & Reset tab table, and the sidebar summary). Update all of them
  together or they'll disagree with each other.
- [ ] **Pipeline flowchart** (`src/pyadps/pipeline_flowchart.dot`) — if
  this is a new optional pipeline stage, add a node in the matching
  cluster, in the matching position. Render it (`dot -Tsvg ... -o /tmp/x.svg`
  or a Sphinx build) to confirm it lays out sanely before committing.
- [ ] **`docs/source/webapp/index.md`** — describe the new control(s).
- [ ] **Test-side mock sync.** Page tests exercise a local `MockXRunner`
  class standing in for the real Runner, so `proc.get_x_runner()` stays
  fast and isolated. If the mock's method surface doesn't include the new
  method, no test ever calls it and the real integration point goes
  completely unexercised, even at 100% suite pass — this happened here
  (`trim_depths()` was missing from `MockVelocityCheckRunner` for a full
  round of "add a feature" work). Add the method to the mock the same PR
  you add it to the page.
- [ ] **Seed new session-state keys everywhere a session-state dict is
  hand-built**, not just the shared `_full_ss()`-style helper — this file
  has several standalone `ss = {...}` dicts in different test classes
  (coverage-gap tests, resampler tests) that don't route through the
  shared helper and will `KeyError` on a rerun if a new key isn't in them
  too.
- [ ] **AppTest gotchas** when writing the new tests:
  - Simulate real interaction with `.check()` / `.uncheck()` / `.set_value(x)`
    on the widget, not by writing directly into `session_state[key]` —
    direct writes don't reflect what a real widget interaction does to
    *derived* state computed later in the same rerun.
  - To test an "uncheck" transition specifically, seed the *pre*-state
    (checked) in the initial session-state dict passed to `AppTest`, then
    do the uncheck as the test's one interaction. Two sequential
    `.set_value(True)` then `.set_value(False)` calls on the same
    checkbox in one test did not reliably stick in this AppTest version —
    confirmed independently on an untouched, pre-existing checkbox, so
    it's a harness quirk, not a page bug; don't chase it as one.
  - If the new chart/check operates on arrays that could realistically
    exceed the `plotly_resampler` threshold (~5000 points) for a real
    deployment, test it with the resampler actually enabled — the
    suite's usual fixtures force `HAS_RESAMPLER = False` for speed, which
    means the resampler code path can silently go untested. A
    `fill="toself"` trace built from a folded/reversed x-array crashed
    here (`FigureResampler` requires strictly monotonic x on every
    trace) and nothing caught it until a real large file hit it.
- [ ] Run the **full suite** (`python -m pytest -q --no-cov`) and a
  **Sphinx build** (`sphinx -b html docs/source /tmp/check -q`) clean
  before calling it done — not just the one test file you were editing.
