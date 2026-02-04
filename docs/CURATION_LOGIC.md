# Curation Logic — Snellen & JCC (dot) charts

This document summarizes the decision logic implemented in `curate_conversations.py` for Snellen and JCC (dot/flip) interactions, plus the requested JCC intent wording corrections.

**1. Overview (row processing order)**
- Normalize `Chart_Display` and `Occluder_State`.
- `get_question()` routes by chart and occluder (JCC occlusion → `INTERMITTENT ERROR`, Flip1/Flip2 use Flip questions).
- `get_answer()` dispatches:
  - Flip rows → `get_flip_answer()`
  - Snellen rows → `get_snellen_answer(current, prev, nxt, prevprev)`
  - JCC rows → `answer_jcc` (or `INTERMITTENT ERROR` for occluded)
  - Others → configured fallbacks

**2. Snellen parsing**
- Chart names like `snellen_chart_40_30_25_40` are parsed by `get_snellen_base_and_highlight()` into:
  - base: `snellen_chart_40_30_25`
  - highlight: `40.0` (imperial system, numeric highlighted optotype)
- The last numeric token is the highlighted line; the base is everything before it.
- **Metric-to-Imperial Conversion:** All Snellen charts are converted to the imperial (20/x) system for consistent comparison, using the mapping:
  - Metric 6/60 → Imperial 20/200 (legally blind)
  - Metric 6/30 → Imperial 20/100
  - Metric 6/20 → Imperial 20/70
  - Metric 6/15 → Imperial 20/50
  - Metric 6/12 → Imperial 20/40
  - Metric 6/9 → Imperial 20/30
  - Metric 6/7.5 → Imperial 20/25 (red line)
  - Metric 6/6 → Imperial 20/20 (normal vision)
  - Metric 6/5 → Imperial 20/16
  - Metric 6/4 → Imperial 20/13
  - Metric 6/3 → Imperial 20/10
- Chart names with metric values are automatically detected and converted. For example:
  - `snellen_chart_60_30_20_6` → highlight `6` detected as metric, converted to imperial `20`
  - `snellen_chart_60_30_20_7_5` → decimal notation `7_5` is recognized as metric `7.5`, converted to imperial `25`
  - `snellen_chart_40_30_25_40` → highlight `40` detected as imperial (not in metric set), used as-is
- **Repeated-value highlight detection:** If the last token in a chart name matches any of the previous tokens (excluding the prefix `snellen_chart`), it is treated as the highlighted optotype and the base excludes the last token. For example:
  - `snellen_chart_20_15_15` → base `snellen_chart_20_15`, highlight `15` (metric 6/15 → imperial 20/50)
  - `snellen_chart_70_60_50_60` → base `snellen_chart_70_60_50`, highlight `60` (metric 6/60 → imperial 20/200)
  - Exception: `snellen_chart_20_20_20` does not trigger this rule to avoid ambiguity with the standard 20/20 naming convention.

**Deciphering chart names using `docs/optotype_sizes.md`**
- Naming is inconsistent across sources, so the curation logic uses an "intelligent deciphering" strategy that references `docs/optotype_sizes.md` to interpret chart names robustly:
  - **Normalize:** lowercase the name, replace separators (`-`, `/`) with `_`, and strip common suffixes like `chart`, `optotype`, or `line`.
  - **Tokenize numbers:** extract all numeric tokens (integers or decimals). If multiple numbers appear, prefer the last numeric token as the highlighted optotype but validate it against known size sets from `docs/optotype_sizes.md`.
  - **Map by proximity:** when a token doesn't exactly match a known size, pick the nearest value from the documented optotype-size arrays (nearest clinical size), and treat the remaining tokens as the base pattern.
  - **Fallback heuristics:** if no numeric token is found, or tokens clearly indicate a pictorial chart, mark the chart as pictorial and exclude it from echart/snellen handling.
  - **Ambiguous sequences:** when several numeric sequences could be the highlight, prefer the sequence that yields a valid base (a repeated pattern of optotype sizes) or the last token when in doubt.
- Examples: `snellen_chart_40_30_25_40` -> base `snellen_chart_40_30_25`, highlight `40`; `snellen-20x` -> highlight `20`.
- Maintain `docs/optotype_sizes.md` with any new observed size-sets to improve automatic mapping accuracy over time.

**3. Snellen decision priority (top → bottom)**
- If no previous snellen row: if highlighted == 20 → `Able to read.` else `Blurry.`
- If previous exists and same base:
  - If current_highlight < previous_highlight → `Able to read.` (finer line)
  - If current_highlight > previous_highlight → `Unable to read.` (coarser line)
  - If highlights equal:
    - If SPH changed between previous and current → normally `Getting better.`
    - If SPH unchanged → `Blurry.`
    - Exception: if SPH oscillation detected across prevprev→prev→current (change then reversal) → `Unable to read.` (avoids false improvement)
- Lookahead (next row) confirmation overrides/affirms transitions:
  - If next row shares same base and next_highlight < current_highlight → `Able to read.` (confirmed)
  - If next_highlight > current_highlight → `Unable to read.` (confirmed regression)
  - If next_highlight == current_highlight and SPH changed between current and next → treat as regression (`Unable to read.`)
  - **Cross-base lookahead:** If next row is a Snellen chart with a finer highlight (smaller value) regardless of base, treat current row as `Able to read.` This handles scenarios where the optometrist transitions to a different chart after confirming the patient can read the current line.
- Fallback: if highlighted == 20 → `Able to read.` else `Blurry.`

**4. SPH change detection**
- `has_sph_change()` compares `R_SPH` and `L_SPH` differences > 0.001.
- Used to detect refinement attempts and to determine `Getting better.` signals.

**5. Oscillation protection**
- If prevprev, prev, current share the same snellen base and highlight, and SPH changed then reversed (prev vs prevprev vs current), the logic returns `Unable to read.` to avoid misclassifying transient SPH flips as improvement.

**6. JCC (dot/flip) handling and corrected intent wording**
- If `Chart_Display == "jcc_chart"` and occluder is `Left_Occluded` or `Right_Occluded`: question and answer are `INTERMITTENT ERROR`.
- `get_flip_answer()` inspects the occluder label to decide Axis vs Power and which eye. It compares current → next phoropter fields to label the intent.

Requested correction to JCC intent wording (applied in documentation):
- Flip 2 (decrease movement) for Right Axis/Power should be labeled as:
  - `Flip 2 - RAM - Red Add Minus` — i.e., Axis/Power decreases in clinical value (examples: power `-0.25` → `-0.50`; axis `175` → `180`).
- Flip 1 (increase movement) should be labeled as:
  - `Flip 1 - GAP - Green Add Plus` — i.e., Axis/Power increases in clinical value (examples: power `-0.25` → `-0.00`; axis `5` → `180`). Exception: a value of `0.00` remains `0.00`.

Notes on the corrected labels and interpretation:
- Flip 2 indicates a decrease in clinical value, append the intent token `Red Add Minus` to indicate the direction and effect (for example, power `-0.25` → `-0.50`, or axis `175` → `180`).
- Flip 1 indicates an increase in clinical value, append `Green Add Plus` (for example, power `-0.25` → `-0.00`, or axis `5` → `180`). Exception: transitions that would produce `0.00` remain `0.00`.
- Example labeling conventions (human-readable):
  - `Flip 2 - RAM - Red Add Minus (axis decreased: 175 -> 180)`
  - `Flip 1 - GAP - Green Add Plus (power increased: -0.25 -> -0.00)`

**7. Flip confidence rules**
- Flip1 rows are always `Confident`.
- Flip2 rows: if any of the next few rows (window defined by `confidence_window_rows` in `conversation_config.json`) show a phoropter or occluder state change, mark as `Confident`. Otherwise default logic (repeat indicates `Confused`).

**8. Snellen confidence rules**
- If next row confirms the same base with a finer highlight (or same highlight + SPH change) → `Confident`.
- **Cross-base confidence:** If next row is a Snellen chart with a finer highlight (smaller imperial value) even across different bases, mark the current row as `Confident`. This recognizes transitions where the optometrist moves to a different chart after validating successful reading.
- If identical question repeats in the configured window with same occluder → `Confused`.
- Special-case SPH oscillation over 3 rows → mark `Confused`.

**9. Implementation notes & file references**
- Code implementing these rules: `curate_conversations.py`
- Config values used: `conversation_config.json` (confidence window, question/answer templates)

---
If you want, I can:
- Apply the *corrected JCC labels* into the code so `get_flip_answer()` returns the precise `Red Add Minus` / `Green Add Plus` suffixes, or
- Produce a per-row annotated CSV that records which rule triggered each row (helpful to audit decisions).

File created: [CURATION_LOGIC.md](CURATION_LOGIC.md)

Next step: mark the todo items complete. 
