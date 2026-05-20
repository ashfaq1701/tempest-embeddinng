# Post-Lock Transition Plan — Clone, Reset, Port, Verify, Archive

**Status:** PENDING. Triggered ONLY after Stage 4 lands AND multi-seed
validates the winning configuration. Do NOT proceed until both gates pass.

**Trigger gates:**
1. Stage 4 (hist_neg_ratio × λ_link sweep, 8 cells) complete.
2. Final locked config identified per v2.4 decision rules.
3. Multi-seed validation (seeds 42, 7, 13) on the winning cell confirms
   stability within ±0.005 across seeds.

**Owner reminder (2026-05-20):** the user laid out this transition
explicitly. Re-read end-to-end before starting Step 1. Do NOT make any
changes in steps 4-5 that weren't pre-approved via port_plan.md review.

---

## Why this way (read once before starting)

The clone-then-port-to-master pattern is safer than cleanup-in-place
because:

1. **Intermediate is frozen.** If you delete something needed, it's still
   in the intermediate, not gone.
2. **Master gets built additively.** Reviewers see clean construction,
   not destructive cleanup. Git history on master tells the story of
   "here's how we built this," not "here's what we removed."
3. **Each port decision is conscious.** Every file moved is a deliberate
   choice, not a default-keep. Forces actual audit, not sweep.
4. **Side-by-side diff debugging.** Any unexpected behavior in master,
   diff the relevant module against the intermediate to find the
   regression.

---

## Step 1 — Freeze the experimental branch as an intermediate clone

Current state: tempest-walk-embedding-new is on
`feature/walk-distribution-embedding`. Clone to a sibling as the
read-only experimental record.

```bash
cd ..
cp -a tempest-walk-embedding-new tempest-walk-embedding-intermediate
cd tempest-walk-embedding-intermediate
git status                # verify branch + dirty state matches source
git log --oneline -20     # verify history is intact

cd ../tempest-walk-embedding-new
git status                # verify we're back in the working tree
```

Confirm clone by diffing a known file:

```bash
diff tempest-walk-embedding-intermediate/scripts/train.py \
     tempest-walk-embedding-new/scripts/train.py
# should be empty
```

If empty diff confirmed, proceed.

**Rule:** Do not commit to or modify the intermediate once cloned. Treat
it as the experimental archive — every ablation, every loss variant,
every dead-end lives there.

---

## Step 2 — Reset tempest-walk-embedding-new to master

```bash
cd tempest-walk-embedding-new
git fetch origin
git checkout master
git pull origin master
git status                # should show clean master at b246b87
```

Verify master is at v3 baseline:

```bash
git log -1 --oneline      # should reference b246b87 or v3 baseline commit
ls tempest_walks/         # should NOT contain timestate.py, normbrake.py, etc.
```

tempest-walk-embedding-new is now fresh master. tempest-walk-embedding-intermediate
holds everything that needs porting.

---

## Step 3 — Produce a port plan BEFORE moving any code

Write `port_plan.md` in tempest-walk-embedding-new categorizing every
file in tempest-walk-embedding-intermediate as:

| Category | Meaning |
|---|---|
| **PORT-DEFAULT** | Part of the locked production config; port and make default in master |
| **PORT-FLAG** | Dead in production but needed for paper ablation reproducibility; port behind a CLI flag (default OFF) |
| **SKIP** | Experimental scaffolding with no production or paper-ablation value; do not port |

Each PORT-DEFAULT and PORT-FLAG entry must include:

- Source path in intermediate
- Destination path in new (usually same)
- CLI flag name (PORT-FLAG only)
- Justification (1 sentence: why this is needed)
- Test plan (how to verify functional correctness)

### Definitive classification (agent decision, 2026-05-20)

Master already has these from b246b87 + recent ports — don't re-port:
- alignment + uniformity loss (default loss family)
- two-table embedding store with cross-table E.1 head
- strict-causal protocol wiring
- TGB Evaluator integration
- TGB negatives, training reservoir, historical/random mix
- `lambda_normbrake`, `normbrake_threshold` (b062674)
- `weight_decay_link` (de70b73)
- All md docs (02041aa)

### PORT-DEFAULT (must port; goes into locked production defaults)

| Item | Source | Justification |
|---|---|---|
| `timestate.py` (NodeTimeState) | feature/`tempest_walks/timestate.py` | Phase 0.5 anchor; Component 0 is the reason wiki anchor hit 0.7079 |
| `TimeEncoder` class | feature/`tempest_walks/model.py` | Component 0 functional time encoding |
| `LinkPredictor` Component 0 input path | feature/`tempest_walks/model.py` | wires Φ(Δt) + cold-start bits into the link MLP |
| `use_time_encoding`, `time_enc_k`, `cold_start_dt_clamp_factor` config fields | feature/`tempest_walks/config.py` | Component 0 plumbing; all default-ON |
| Early stopping + patience | feature/`tempest_walks/trainer.py` | training infrastructure; needed for anchor validation gate |
| Best-weight snapshot/restore | feature/`tempest_walks/trainer.py` | needed for `best_val_mrr` reporting |
| `monitor_sample_pct`, `skip_final_full_eval` | feature/`scripts/train.py` + trainer | review-scale eval infrastructure (review is 30× wiki) |
| `--log-debug` per-epoch instrumentation | feature/`tempest_walks/trainer.py` | needed to diagnose future cliffs |
| `scripts/anchor_validation.py` (3-seed driver) | feature/scripts/ | the gate for porting correctness |
| Phase 0.5 diagnostic scripts | feature/scripts/ | cold-start coverage check; reproducible |

### PORT-FLAG (paper-ablation paths; CLI flag, default OFF/locked)

| Item | CLI flag | Default | Reason needed for paper |
|---|---|---|---|
| Alternative loss families | `--primary-loss {triplet,infonce,sgns}` | `alignment` | Table 4.7 wiki sweep + Table 4.3 review sweep both show alignment wins/ties |
| Triplet hyperparameters | `--triplet-margin`, `--weight-decay-emb` | 0.5, 1e-4 | inert unless `--primary-loss=triplet` |
| InfoNCE hyperparameters | `--infonce-tau`, `--infonce-num-neg-{in-batch,unif}` | 0.1, 256, 256 | inert unless `--primary-loss=infonce` |
| SGNS hyperparameters | `--sgns-k-neg`, `--sgns-subsample-t`, `--sgns-lr-*` | Mikolov defaults | inert unless `--primary-loss=sgns` |
| E.2 head variant | `--head-mode component_0_only` | `cross_table` (E.1) | Phase S settled E.1 wins; paper shows the comparison |
| A2-off (no alignment) | `--lambda-align 0` | 1.0 | Phase S settled A2-on wins; paper shows the comparison |
| `hist_neg_ratio` variants | `--hist-neg-ratio` | (Stage 4 winner — likely 0.5) | TGB-distribution match; paper-defensible ablation |
| `eta_uniform`, `uniformity_cap` | `--eta-uniform`, `--uniformity-cap` | (Stage 5 winners) | uniformity sweep is an open ablation |
| `lambda_normbrake`, `normbrake_threshold` (off) | already ported | locked ON (0.1, dataset threshold) | run-with-no-cliff-fix ablation |

### SKIP (do not port — config or code that's been demonstrated to hurt or has no paper value)

User directive 2026-05-20: "don't port configs that don't matter." Even
though "PORT-FLAG when unsure" is the general rule, configs that ALWAYS
LOSE in the sweeps should be SKIPPED — they're clutter, not
paper-defensibility.

| Item | Why SKIP |
|---|---|
| `align_weighting` (B / C variants) | Phase 1 ablation completed; A won unambiguously. Keep ONLY the A behavior, drop the variants. |
| `cross_table_dropout` | Stage 2 hurt the cliff; never useful. |
| `link_mlp_n_layers` (n=5 variant) | Stage 2 HURT; reaffirms Lesson 15 ("capacity hurts on wiki"). Keep n=3 hard-coded. |
| `link_mlp_dropout` | Stage 2 marginal-to-hurt; doesn't address embedding-side cliff. |
| `embedding_dropout` | Stage 2 hurt; makes cliff sharper. |
| `lambda_link` (joint training) | Stage 3 DECISIVELY FALSIFIED (Lesson 19). No paper question needs this knob — the paper documents the falsification in narrative. SKIP CLI entirely. |
| Per-walk dispatch helpers (if any) | leftover from walks-encoder branch; verify usage = 0 before deleting |
| InfoNCE NaN-debug instrumentation | bug fixed, prints removed |
| Stage 2/3/4/5 sweep wrapper scripts | one-off experiment drivers; reproducible from doc + the underlying train.py |
| Loss-search amendment doc archive | superseded by v2.4 §9 + Lesson 17–22 |
| EdgeBank-distillation code (if started) | removed in amendment v1.2 |
| tempest_walks/walks-encoder variants | different branch; do not bring over |

### Rule of thumb (codified)

The PORT-FLAG vs SKIP test:
1. **PORT-FLAG** ⟺ the paper needs to SHOW the result (e.g., "Triplet on review", "E.2 head"). The flag enables reproducing the ablation.
2. **SKIP** ⟺ the knob is demonstrably-bad AND no paper-defensible question needs the flag (e.g., "λ_link collapses immediately" — paper narrates, doesn't need CLI to reproduce).

When in doubt: a single CI run + a sentence in the paper > a CLI flag that no one will use again.

### SKIP list (no production use, no paper-ablation value)

User directive 2026-05-20: **don't port configs that don't matter.**
Even though the general rule is "default to PORT-FLAG when unsure,"
configs and code paths that are DEMONSTRATED to hurt should be SKIPPED
entirely — they don't need to live on master even as flags.

**Config fields to SKIP** (demonstrated not useful, not paper-defensible):

- `align_weighting` (A/B/C variant in alignment_loss) — A won and is the
  current default; B and C never helped. SKIP the variant code.
- `cross_table_dropout` — Stage 2 hurt (E.3 ablation). SKIP.
- `link_mlp_n_layers` — Stage 2 hurt (deeper = worse on wiki, Lesson 15
  reaffirmed). SKIP.
- `link_mlp_dropout` — Stage 2 marginal-to-hurt. SKIP.
- `embedding_dropout` — Stage 2 hurt. SKIP.
- `lambda_link` — Stage 3 DECISIVELY FALSIFIED (val MRR collapses
  immediately at any λ_link > 0). The mechanism is documented in Lesson
  19. No paper ablation needs this — it's a known-bad knob. SKIP from
  master entirely.
- `weight_decay_emb` — only ever fired under `primary_loss=triplet`;
  default-disabled. If Triplet ports as PORT-FLAG, this comes with it;
  otherwise SKIP.

**Code paths to SKIP**:

- Stage 2 architectural variants (deeper MLP, embedding dropout — these
  are hyperparameters that didn't help; no code path needs to live).
- InfoNCE NaN-debug instrumentation (bug fixed, debug printouts gone).
- One-off scripts from intermediate stages that aren't anchor or
  diagnostic.
- EdgeBank distillation code if ever started (removed in amendment v1.2).
- tempest_walks variants from the walks-encoder overnight session
  (different branch).
- Per-walk dispatch kernel + helpers if all walk-encoder bypasses are
  removed (verify usage before porting).

**Rule for ambiguous configs**: if the config has a default that ALWAYS
WINS in Stage 2/3/Stage S sweeps, port ONLY the default behavior (not
the knob). Adding a CLI flag for "the value that always loses" is
clutter, not paper-defensibility.

**The PORT-FLAG vs SKIP test**: a PORT-FLAG is justified when the paper
needs to SHOW the result (e.g., "Triplet vs alignment on review — we
ran both"). A SKIP is justified when the knob is demonstrably-bad and
no paper-defensible question needs the flag (e.g., "we proved λ_link
collapses, no reason to keep the CLI flag").

**Commit `port_plan.md` to tempest-walk-embedding-new master and pause
for review before porting any code. Do not start Step 4 until plan is
approved.**

---

## Step 4 — Execute the port in reviewable chunks

Once port_plan.md is approved, execute the port as small, reviewable
commits. NOT one massive merge. Each commit:

- Single feature or module
- Includes tests for that module
- Includes CLI flag plumbing if PORT-FLAG
- Includes doc update if user-facing

### Recommended commit ordering (small to large dependencies)

1. Port `tempest_walks/timestate.py` (no upstream deps)
2. Port `model.py` changes for TimeEncoder + extended LinkPredictor
3. Port `scripts/train.py` CLI knobs (one knob per commit)
4. Port normbrake module
5. Port alternative losses (one commit per loss, behind `--primary-loss`)
6. Port E.2 head variant behind `--head-variant` flag
7. Port diagnostic scripts
8. Port anchor-validation script
9. Update Config defaults to lock production values
10. Add `config_locked_v1.yaml` with locked production hyperparameters

### Each commit must pass

- Unit tests for the ported module
- 2-epoch smoke test confirming end-to-end training works
- For PORT-FLAG commits: smoke test in the non-default flag setting

### Final verification (after all commits land)

Run anchor validation (3 seeds, 2 epochs) on new master. Result MUST
match original anchor (test MRR **0.7070 ± 0.0016**) within anchor std.
If it doesn't, the port has a bug.

---

## Step 5 — Archive the intermediate

Once master is verified:

```bash
cd ..
git -C tempest-walk-embedding-intermediate gc --aggressive
tar czf tempest-walk-embedding-intermediate-archive-$(date +%Y%m%d).tar.gz \
    tempest-walk-embedding-intermediate/

# Keep the directory in place for now. Don't delete.
```

The tar archive goes to long-term storage. Directory stays as working
reference while paper ablations run. Once paper is submitted and
ablations finalized, the directory can be removed but the tar stays.

---

## Deliverables of this Step-1-through-5 process

1. `tempest-walk-embedding-intermediate/` — frozen experimental record
2. `tempest-walk-embedding-new/` at master with:
   - Locked production config as defaults
   - All paper-ablation paths behind CLI flags
   - Reproduces anchor result (0.7070 ± 0.0016) on 3-seed validation
   - `port_plan.md` committed
   - `config_locked_v1.yaml` committed
   - Codebase audit reviewable from git log
3. `tempest-walk-embedding-intermediate-archive-YYYYMMDD.tar.gz`

---

## Estimated time

For a human reviewer/operator (reading-and-deciding pace), this is 2–3
days of focused work. For an agent operating the tooling directly, it
is minutes-to-low-hours.

| Step | Human time | Agent time |
|---|---|---|
| Step 1 — clone | 10 min | ~30 sec (cp -a + git verify) |
| Step 2 — master reset | 10 min | ~30 sec (checkout + verify) |
| Step 3 — port plan | 4–6 hr | ~15–30 min (read intermediate file-by-file + classify) |
| Step 4 — execute port | 1–2 days | ~1–2 hr (small commits + tests + anchor) |
| Step 5 — archive | 30 min | ~30 sec (tar) |
| **Total** | **2–3 days focused work** | **~2–3 hours** |

Critically, the anchor-validation step (50-ep × 3 seeds = ~3 hr GPU
time) is the binding constraint either way — not the tooling work.

---

## Rules of thumb during the port

- If unsure between PORT-FLAG and SKIP, **default to PORT-FLAG**. Adding
  is cheap; resurrecting deleted code is expensive.
- If intermediate had tests for a function, the port MUST have those
  tests. If it had no tests, the port must add them.
- Anchor validation is ground truth for "is the port correct." If it
  doesn't reproduce 0.7070 ± 0.0016, something is wrong. Don't dismiss
  small drifts as CUDA noise — Stage 3 showed even Adam constructor
  changes cause 0.030 drift. Be suspicious of anything > 0.005.
- Commit `port_plan.md` BEFORE moving code. Don't decide what to port
  as you port.

---

## When to pause and ask for direction during porting

- If `port_plan.md` classifications are ambiguous (PORT-FLAG vs SKIP)
- If anchor validation fails after the port and the bug isn't found in
  30 minutes
- If a PORT-FLAG path has dependencies on PORT-DEFAULT code that's
  been refactored, and resolving requires non-trivial redesign
- If something clearly important is on the intermediate but missing from
  the port plan → **update this document and pause.**

---

## DO NOT

- Proceed to architecture-sweep branch creation until anchor validation
  on new master passes.
- Skip the port_plan.md review gate.
- Bundle the port into one big commit.
- Treat anchor drift > 0.005 as acceptable noise.
- Delete the intermediate directory before paper is submitted.
