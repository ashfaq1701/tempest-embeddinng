# Port plan — feature/walk-distribution-embedding → master

**Source:** `tempest-walk-embedding-intermediate/` (frozen experimental branch).
**Target:** `tempest-walk-embedding-new/` (master).
**Locked production config:** see v2.4 §1.

**Status:** WRITTEN 2026-05-21. Pause for user review BEFORE Step 4 execution.

---

## Already on master from prior commits (skip — don't re-port)

| File | Source commit |
|---|---|
| `walk_distribution_matching_embedding_v2.md`, `_v2.4_skeleton.md` (WIP), `_results.md`, `loss_function_search_ammendum.md`, `post_lock_transition_plan.md` | `02041aa` |
| `CLAUDE.md` Lessons 17–22 | `02041aa` |
| `tempest_walks/losses.py::normbrake_loss` | `b062674` |
| `tempest_walks/config.py::lambda_normbrake`, `normbrake_threshold` | `b062674` |
| `tempest_walks/trainer.py` normbrake wiring in `_embedding_step` | `b062674` |
| `scripts/train.py` `--lambda-normbrake`, `--normbrake-threshold` | `b062674` |
| `tempest_walks/config.py::weight_decay_link` | `de70b73` |
| `tempest_walks/trainer.py` link_optimizer weight_decay | `de70b73` |
| `scripts/train.py` `--weight-decay-link` | `de70b73` |

Master is at `de70b73` (3 commits ahead of `b246b87`).

---

## PORT-DEFAULT (locked production code)

### A. Component 0 (time encoding + cold-start bits) — PRIMARY GAIN over v3 baseline

| Item | Source path | Target path | Test plan |
|---|---|---|---|
| `NodeTimeState` class | intermediate/`tempest_walks/timestate.py` | new file `tempest_walks/timestate.py` | unit test: observe(src, tgt, t) updates last_seen_dst/src tables; query returns correct Δt + cold-start bits |
| `TimeEncoder` class (Φ functional time encoder) | intermediate/`tempest_walks/model.py` | merge into `tempest_walks/model.py` | unit test: Φ(0) ≈ 0; Φ(time_scale) ≈ basis at frequency 1 |
| LinkPredictor Component 0 input slot | intermediate/`tempest_walks/model.py::LinkPredictor` | merge into `tempest_walks/model.py::LinkPredictor` | smoke test 2 ep w/ + w/o `--use-time-encoding`; with on must reach val ≥ 0.70 on wiki ep 1 |
| Config fields: `use_time_encoding`, `time_enc_k`, `cold_start_dt_clamp_factor` | intermediate/`tempest_walks/config.py:22-24` | `tempest_walks/config.py` | smoke test |
| Trainer integration: time_state instance, observe at post-scoring, query at link forward | intermediate/`tempest_walks/trainer.py` | merge into `tempest_walks/trainer.py` | 2-ep anchor run reproduces 0.7070 ± 0.0016 on wiki |
| CLI: `--no-use-time-encoding` BooleanOptionalAction | intermediate/`scripts/train.py` | `scripts/train.py` | smoke test in OFF mode |

**Defaults:** `use_time_encoding=True`, `time_enc_k=16`, `cold_start_dt_clamp_factor=100.0`.

### B. Training infrastructure

| Item | Source path | Target path | Test plan |
|---|---|---|---|
| Early stopping + patience | intermediate/`tempest_walks/trainer.py` | merge into `tempest_walks/trainer.py` | 2-ep smoke with patience=1 |
| Best-weight snapshot/restore | intermediate/`tempest_walks/trainer.py` | merge into trainer | 2-ep smoke: weights at end of training match snapshot from best epoch |
| `monitor_sample_pct`, `skip_final_full_eval` | intermediate/`tempest_walks/{evaluator,trainer}.py` + `scripts/train.py` | merge into all three | smoke test with `--monitor-sample-pct 0.1` |
| `--log-debug` per-epoch instrumentation (col_norm, link_w_norm, grad_E_target, grad_E_context, L_normbrake) | intermediate/`tempest_walks/trainer.py` | merge into trainer | smoke test; verify log keys present |
| CLI: `--early-stop-patience`, `--num-epochs`, `--monitor-sample-pct`, `--skip-final-full-eval`, `--log-debug` | intermediate/`scripts/train.py` | `scripts/train.py` | argparse smoke |

### C. Tooling

| Item | Source path | Target path | Test plan |
|---|---|---|---|
| Anchor validation driver | intermediate/`scripts/anchor_validate.py` | `scripts/anchor_validate.py` | run on master: 3 seeds × 2 ep, mean test MRR == 0.7070 ± 0.0016 |
| Phase 0.5 diagnostic | intermediate/`scripts/phase0_5_diag.py` | `scripts/phase0_5_diag.py` | run on master: cold-start prevalence == 99.1% on wiki |
| Init divergence check | intermediate/`scripts/init_divergence_check.py` | `scripts/init_divergence_check.py` | run on master: across-seed init divergence within tolerance |

### D. Locked config defaults (already on master from earlier ports + new updates)

Need to flip in `tempest_walks/config.py`:
- `lambda_normbrake: 0.0` → `0.1` (locked production)
- `normbrake_threshold: 0.0` → leave at 0.0 (calibrate per dataset in entry script; or read from a per-dataset table)
- `weight_decay_link: 0.0` → `1e-4` (locked production)

Alternatively keep defaults OFF and require explicit CLI — but then anchor reproduces with defaults OFF (v3 baseline behavior), which is what Gate A actually tests. **Keep defaults OFF**; require explicit CLI to engage locked production. Document in CLAUDE.md.

---

## PORT-FLAG (paper-ablation paths, CLI default OFF)

### E. Alternative loss families

| Item | Source path | Target path | CLI | Test plan |
|---|---|---|---|---|
| `triplet_loss` + semi-hard mining helpers | intermediate/`tempest_walks/losses.py` | merge into `tempest_walks/losses.py` | `--primary-loss triplet` | 2-ep wiki run: train completes, no NaN |
| `infonce_loss` + logsumexp denominator | intermediate/`tempest_walks/losses.py` | merge | `--primary-loss infonce` | 2-ep run; no NaN at τ=0.1 |
| `sgns_loss` + unigram^0.75 cache + Mikolov lr schedule | intermediate/`tempest_walks/losses.py` + trainer | merge | `--primary-loss sgns` | 2-ep run; lr decays correctly |
| `_positional_weights` shared helper | intermediate/`tempest_walks/losses.py` | merge | (internal) | shared by all 3 |
| Config fields: `primary_loss`, `triplet_margin`, `weight_decay_emb`, `infonce_*`, `sgns_*` | intermediate/`tempest_walks/config.py:84-109` | merge | CLI flags | argparse smoke |
| Trainer dispatch on `primary_loss` in `_embedding_step` | intermediate/`tempest_walks/trainer.py` | merge | (implicit via flag) | each loss runs 2 ep |

### F. Head variants

| Item | Source path | Target path | CLI | Test plan |
|---|---|---|---|---|
| `head_mode="component_0_only"` (E.2) | intermediate/`tempest_walks/model.py::LinkPredictor` | merge | `--head-mode component_0_only` | 2-ep run; reproduces Phase S E.2+A2-off 0.7079 result |

### G. A2-off (no alignment) ablation

| Item | Source path | Target path | CLI | Test plan |
|---|---|---|---|---|
| `lambda_align` config field; multiplier in `_embedding_step` | intermediate/`tempest_walks/{config,trainer}.py` | merge | `--lambda-align 0` | 2-ep run with lambda_align=0; A2-off Phase S behavior |

---

## SKIP (demonstrably bad or no paper value)

### Config fields to SKIP

- `align_weighting` (B / C variants in `alignment_loss`) — A won unambiguously in Phase 1.
- `cross_table_dropout` — Stage 2 hurt.
- `link_mlp_n_layers` (n=5 variant) — Stage 2 hurt (Lesson 15 reaffirmed).
- `link_mlp_dropout` — Stage 2 marginal-to-hurt.
- `embedding_dropout` — Stage 2 hurt.
- `lambda_link` — Stage 3 + Stage 4 DECISIVELY FALSIFIED (Lesson 19). Paper documents narrative; no CLI knob needed.

### Code to SKIP

- All `align_weighting` B/C branches in `alignment_loss` and `_positional_weights`.
- Joint training code path in `trainer._link_step` (the `joint = self.config.lambda_link > 0` block).
- Stage 2 architectural variant code in model.py (n_layers parameterization beyond n=3, dropout layers).
- InfoNCE NaN-debug instrumentation (bug already fixed; debug prints removed in port).

### Scripts to SKIP

All sweep wrapper scripts are one-off experiment drivers; reproducible from documented configs:
- `run_alignment_jl.sh`, `run_autonomous_chain.sh`, `run_chain_after_481.sh`
- `run_loss_sweep.sh`, `run_multiseed.sh`, `run_review_sweep_auto.sh`
- `run_section_4_8_1.sh`, `run_section_4_8_2.sh`, `run_section_4_8_3.sh`
- `run_stage2_alignment_fixes.sh`, `run_stage3_lambda_link_wd.sh`, `run_stage4_hist_neg_sweep.sh`, `run_stage5_uniformity_sweep.sh`
- `summarize_runs.py`

---

## Doc sync to master

- `walk_distribution_matching_embedding_v2.4_skeleton.md` master version is WIP; feature is FINAL. **Re-port the FINAL version** as part of Step 4.
- `post_lock_transition_plan.md` master is in sync; no change.
- `CLAUDE.md` master has Lessons 1–22; feature is in sync.

---

## Commit ordering (Step 4 execution)

1. **C1:** sync `walk_distribution_matching_embedding_v2.4_skeleton.md` FINAL from feature.
2. **C2:** port `tempest_walks/timestate.py` (standalone module, no upstream deps).
3. **C3:** port `TimeEncoder` class into `tempest_walks/model.py`.
4. **C4:** port LinkPredictor Component 0 input slot + `head_mode` parameter into `tempest_walks/model.py`.
5. **C5:** port Trainer integration of `time_state` + Component 0 into `tempest_walks/trainer.py`.
6. **C6:** port Config fields for Component 0 + `head_mode` + `lambda_align` into `tempest_walks/config.py`.
7. **C7:** port `scripts/train.py` CLI flags for Component 0 + `head_mode` + `lambda_align`.
8. **C8:** port early-stopping + snapshot/restore + monitor_sample_pct + skip_final_full_eval + --log-debug into Trainer + Evaluator + train.py.
9. **C9:** port `scripts/anchor_validate.py`, `scripts/phase0_5_diag.py`, `scripts/init_divergence_check.py`.
10. **C10 (PORT-FLAG):** port Triplet loss + config fields + CLI flag + trainer dispatch.
11. **C11 (PORT-FLAG):** port InfoNCE loss + config fields + CLI flag.
12. **C12 (PORT-FLAG):** port SGNS loss + unigram cache + Mikolov lr schedule + config fields + CLI flag.
13. **C13:** add `config_locked_v1.yaml` with locked production hyperparameters documented.

After C13: Gates A + B (anchor validation + 50-ep wiki locked-config).

Each commit must pass:
- argparse / import smoke check
- 2-ep wiki smoke run (where applicable)
- For PORT-FLAG: 2-ep run in non-default flag mode (Triplet / InfoNCE / SGNS / E.2 / A2-off)

---

## Estimated time

| Phase | Time |
|---|---|
| C1 — doc sync | 30 sec |
| C2 — timestate.py port | 5 min |
| C3–C7 — Component 0 wiring + CLI | 30 min |
| C8 — training infrastructure | 30 min |
| C9 — diagnostic scripts | 10 min |
| C10–C12 — PORT-FLAG losses | 30 min |
| C13 — config_locked_v1.yaml | 5 min |
| Gate A — anchor validation | 10 min |
| Gate B — 50-ep wiki | 60 min |
| **Total** | **~3 hr** |

---

## Pre-commit cumulative diff sanity

Before C13, run:

```bash
diff -rq /home/ms2420/CLionProjects/tempest-walk-embedding-new \
         /home/ms2420/CLionProjects/tempest-walk-embedding-intermediate \
    | grep -v -E "\.venv|__pycache__|runs/|\.git" | head -30
```

Differences after port should ONLY be:
- SKIP'd items (sweep scripts, dead config fields, demonstrably-bad code paths)
- Doc sync (master's docs are FINAL; intermediate's are WIP)
