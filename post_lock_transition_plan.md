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

### Minimum PORT-FLAG list (these MUST be kept behind flags for paper ablations)

- A2-off code path (no alignment loss)
- normbrake-off code path (`--lambda-normbrake=0`)
- Triplet loss implementation + semi-hard mining
- InfoNCE loss implementation
- SGNS loss implementation + unigram^0.75 cache
- E.2 head variant (Component-0-only head)
- Component 0 disable flag (`--use-time-encoding=False`)
- Different hist_neg_ratio values (CLI knob already exists; verify plumbing)

### PORT-DEFAULT list (subject to Stage 4 outcome — fill in after locking)

- Component 0 (`timestate.py` + TimeEncoder + cold-start bits)
- Alignment + uniformity loss (current v3 implementation)
- Normbrake regularizer at locked threshold per dataset
- λ_link joint training [IF Scenario 1 from Stage 4]
- hist_neg_ratio=0 default [IF Scenario 1 from Stage 4]
- Strict-causal protocol wiring (pre-scoring read, post-scoring write)
- TGB Evaluator integration
- anchor validation script (`scripts/anchor_validation.py`)
- Diagnostic scripts (Phase 0.5 diag, init_divergence_check)

### SKIP list (no production use, no paper-ablation value)

- Stage 2 architectural variants that lost (deeper MLP, embedding dropout
  — these are hyperparameters, not code)
- InfoNCE NaN-debug instrumentation (bug fixed, debug printouts gone)
- One-off scripts from intermediate stages that aren't anchor or
  diagnostic
- EdgeBank distillation code if ever started (removed in amendment v1.2)
- tempest_walks variants from the walks-encoder overnight session
  (different branch)

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
