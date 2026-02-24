# Implementation Coverage vs Plan – Gaps and ADD Test

**Status:** Gaps listed below have been implemented and tested (see test_add_phase.py and test_phase_flow_validation_near.py).

## 1. What’s covered

- **protocol.yaml**: Phases A–Q in the right order (distance_vision → … → binocular_balance → near_add_adjust → near_chart_testing). H, M, N, P, Q defined with triggers, intents, exit_conditions. Chart refs: `snellen_20_20`, `snellen_near_vision`.
- **interactive_session.py**: `phase_names` includes all 17 phases (H, M, N, P, Q). `add_right` / `add_left`, `validation_*_fallback_power`, `validation_*_status`. Processors: `_process_validation_right`, `_process_validation_left`, `_process_validation_distance`, `_process_near_add_adjust`, `_process_near_chart_testing`. Phase routing in `process_response` for these phases.
- **phoropter-ui**: `REFRACTION_PHASES` has 15 main phases A–Q. Phase dropdown (index.html) lists all. `phaseConfigurations` for P and Q with `showADD: true`, chartTab 5. ADD row in refraction table; `adjustValue` and table mousedown support `add`; ADD shown only in P/Q via `handlePhaseChange`. Session has `subjective.R.add` / `subjective.L.add`. Export/CSV includes ADD.
- **Styles**: `.add-row`, `.add-input` in styles.css.

---

## 2. What’s missing or wrong

### 2.1 Phase flow (critical)

- **Duochrome → validation**
  - After **duochrome_right** “Both Same” / reversal the code calls `_transition_to_left_eye_refraction()`. Per plan it should go to **Phase H (validation_right)**.
  - After **duochrome_left** “Both Same” / reversal the code calls `_transition_to_binocular_balance()`. Per plan it should go to **Phase M (validation_left)**.
- **Required**
  - Add `_transition_to_validation_right()` (from duochrome_right): set phase, occluder Left_Occluded, chart 20/20, set_power, jcc_control("R").
  - Add `_transition_to_validation_left()` (from duochrome_left): set phase, occluder Right_Occluded, chart 20/20, set_power, jcc_control("L").
  - Add `_transition_to_validation_distance()` (from validation_left “Yes”): BINO, chart 20/20.
  - Add `_transition_to_near_add_adjust()` (from binocular_balance complete): BINO, Chart5.
  - In `_process_duochrome_right`: on “Both Same” / reversal call `_transition_to_validation_right()` instead of `_transition_to_left_eye_refraction()`.
  - In `_process_duochrome_left`: on “Both Same” / reversal call `_transition_to_validation_left()` instead of `_transition_to_binocular_balance()`.
  - In `_process_validation_right` “Yes”: call `_transition_to_left_eye_refraction()`.
  - In `_process_validation_left` “Yes”: call `_transition_to_validation_distance()`.
  - In `_process_validation_distance` “Yes”: call `_transition_to_binocular_balance()`.
  - In binocular_balance completion: call `_transition_to_near_add_adjust()` instead of going to “complete” (if P/Q are in scope).

### 2.2 _determine_next_phase

- Still has: `duochrome_right → left_eye_refraction`, `duochrome_left → binocular_balance`, `binocular_balance → complete`.
- Update to:  
  `duochrome_right → validation_right`, `validation_right → left_eye_refraction`, `duochrome_left → validation_left`, `validation_left → validation_distance`, `validation_distance → binocular_balance`, `binocular_balance → near_add_adjust`, `near_add_adjust → near_chart_testing`, `near_chart_testing → complete`.

### 2.3 _setup_phase

- No branches for: `validation_right`, `validation_left`, `validation_distance`, `near_add_adjust`, `near_chart_testing`.
- “Jump to Phase” for H, M, N, P, Q does not set chart/occluder/power. Add:
  - **validation_right**: occluder Left_Occluded, chart 20/20 (see 2.5), set_power, jcc R.
  - **validation_left**: occluder Right_Occluded, chart 20/20, set_power, jcc L.
  - **validation_distance**: occluder BINO, chart 20/20, set_power, jcc BINO.
  - **near_add_adjust**: BINO, Chart5 (near chart), set_power (and ADD if API supports it).
  - **near_chart_testing**: same as P, chart_5.

### 2.4 20/20 validation fallback (Better / Same / Worse)

- Backend sets `fallback_active` and returns intents “Better”, “Same”, “Worse” but:
  - Does **not** switch chart to 20/40 (no `set_chart` for 20/40).
  - Does **not** handle a second response (“Better”/“Same”/“Worse”): no SPH refinement loop, no revert to `validation_*_fallback_power`, no transition to next phase.
- Add:
  - When entering fallback: call `set_chart` for 20/40 (chart_13 + size `"40"` per API; plan’s chart_12 is 70/60/50, so use chart_13 for 20/40).
  - Sub-state or flag so that when in validation_right/left with `fallback_active`, the next intent is “Better”/“Same”/“Worse”.
  - “Better”: apply −0.25 D SPH to that eye (max 3–4 steps), then re-ask or accept and go to next phase.
  - “Same”: accept current power, next phase.
  - “Worse”: revert that eye to `validation_*_fallback_power` (set_power_with_prev_state), then next phase.
  - After resolution, clear fallback state and move to next phase (left_eye_refraction from H, validation_distance from M).

### 2.5 20/20 chart name

- Protocol uses `charts: ["snellen_20_20"]`. `chart_map` has `snellen_chart_20_20_20` → chart_15 (no `snellen_20_20`).
- So either add `"snellen_20_20": "chart_15"` to `chart_map` and use size `"20_1"` (or 20_2/20_3) when calling set_chart with size, or use `snellen_chart_20_20_20` in transitions/setup. API 20/20 is chart_15 with 20_1/20_2/20_3; chart_14 is 20/15/10.

### 2.6 ADD in API (run-tests and responses)

- `set_power` and `set_power_with_prev_state` do **not** send `add` in `right_eye`/`left_eye`. So when Phase P does `add_right += 0.25` and calls `set_power_with_prev_state`, the phoropter never receives ADD.
- `_build_response()` and API server `get_status` do **not** include `add` in `power.right`/`power.left`. Frontend cannot show backend ADD in “current power” from API.
- Add:
  - Optional `r_add`/`l_add` (or `add` in each eye) in `set_power` and `set_power_with_prev_state` payloads when provided; pass through to run-tests so ADD can be tested on hardware.
  - In `_build_response()`, add `"add": self.add_right` and `"add": self.add_left` (or from current_row if you sync r_add/l_add from add_right/add_left) to `power.right` and `power.left`.
  - In `api_server` `get_status`, add `"add": session.add_right` and `session.add_left` (or from current_row) to `current_power.right` and `current_power.left`.

### 2.7 protocol.yaml duplicate id

- `validation_left` has `id: "M"` and `detection_failed` also has `id: "M"`. Change `detection_failed` to e.g. `id: "M_QA"` or `id: "DETECT_FAIL"` to avoid duplicate.

### 2.8 Validation / fallback message in UI

- Plan: “Add message container for fallback/validation messages”. No `.validation-message` or similar in HTML/JS; backend returns `"message": "20/20 not clear. Switched to 20/40..."` but frontend does not show it. Add a small message area (e.g. under the phase selector or above intents) and display `response.message` when present (e.g. when `status === 'fallback_active'`).

### 2.9 ADD range (optional)

- Plan: ADD 0.00–3.50 D. `adjustValue` for `add` does not clamp. Optionally clamp ADD in backend (Phase P) and/or frontend to 0–3.50.

---

## 3. ADD test

- **Run-tests with ADD**: Use the curl in the repo (see below) to confirm the phoropter backend accepts `add` in `right_eye`/`left_eye`. If it does, add `add` to `set_power`/`set_power_with_prev_state` as in 2.6.
- **Session ADD state**: A small test (e.g. in `eye_test_engine/tests/test_add_phase.py`) can start a session, jump to `near_add_adjust`, send “Blurry” a few times, then “Comfortable”, and assert `session.add_right`/`session.add_left` and that the last response includes `current_add` or power with add.

See `scripts/test_add_run_tests.sh` for the curl test.
