---
type: pre-registration
probe-id: probe-0
status: locked
date-locked: 2026-05-13
date-run: 
date-complete: 
gates: [probe-1, probe-0a, probe-0b, probe-0c, probe-0d, probe-0e]
linked-paper: icra2027-paper
tags: [experiment, dual-system-vla, retrieval, cluster-analysis]
---

> **LOCKED 2026-05-13.** Changes to §3 (Hypotheses) and §4 (Thresholds) require an entry in §12 (Amendments Log) with timestamp and justification before they can be applied to results interpretation.

# Probe 0 — Cluster Structure of Dual-System VLA Hidden States

## 1. Research Program Context

This probe is the earliest falsification test for a larger research program on **retrieval-augmented inference for dual-system Vision-Language-Action models**.

The program's central thesis is that the slow System-2 component of dual-system VLAs (a 2–7B vision-language model) can be replaced or bypassed at inference time by retrieving precomputed latent representations from a memory bank, indexed by instruction and visual context. If successful, this would significantly reduce inference latency without sacrificing task performance. The target publication venue is ICRA 2027 with optional earlier CoRL 2026 workshop submission.

The thesis rests on one structural assumption: that the latent representations produced inside a dual-system VLA are *semantically structured* in a way that supports nearest-neighbor retrieval — i.e., similar tasks produce similar latents under a metric usable by an approximate-nearest-neighbor index.

This probe tests that assumption on its own, with no downstream architecture changes. The probe answers a binary question (does the structure exist?) that gates the next four months of work. If the assumption fails, the program needs reformulation before further implementation; if it holds, this probe provides the motivating figure and methods-section justification for the paper.

## 2. Models and Data

### 2.1 Model under probe

- **Identifier:** `lerobot/pi05_libero_base`
- **Origin:** Physical Intelligence's π0.5, fine-tuned on LIBERO, ported to LeRobot format
- **Architecture:** PaliGemma-3B VLM (System 2) + ~315M flow-matching action expert (System 1) connected via cross-attention in a Mixture-of-Transformers layout
- **Precision:** bfloat16
- **Forward path used:** Training-path fused prefix+suffix forward (the path that uses `embed_prefix` + `embed_suffix` and a single joint forward through `paligemma_with_expert`). Not the inference-time two-forward path.

**Why this model:**
The openpi codebase exposes both prefix (VLM-side) and suffix (expert-side) hidden states cleanly. This gives two parallel embedding streams to compare as potential retrieval keys. π0.5 is also a current-SOTA flow-matching dual-system VLA, so positive results transfer to production-relevant systems.

**Caveat regarding model choice:**
π0.5's hidden states are not specifically optimized to be retrieval-friendly. By contrast, OpenHelix uses a learned `<ACT>` token explicitly trained as a single extraction query, which is closer in spirit to what a retrieval index would want. A *negative* result on π0.5 in this probe does **not** falsify the broader research program — it specifically falsifies the weaker claim that off-the-shelf π0.5 hidden states are retrieval-friendly. A follow-up probe on OpenHelix `<ACT>` is required regardless of this probe's outcome.

### 2.2 Dataset

- **Identifier:** `lerobot/libero`
- **Format:** LeRobot
- **Scope:** Full LIBERO benchmark, 130 tasks across 4 suites (Spatial, Object, Goal, Long)
- **Used as source of:** Both observations (RGB + wrist + state) and instructions
- **Frames sampled:** `N_SAMPLES = 800` base frames; with `NOISE_DRAWS_PER_FRAME = 3` this yields ~2400 embeddings per stream
- **Sampling seed:** `42`

**Caveat regarding data choice:**
The fine-tuned checkpoint has seen these exact LIBERO instruction strings during training. Cluster quality on this probe may reflect memorization rather than generalization. The probe does *not* test out-of-distribution generalization; that is a scoped follow-up (see §10).

## 3. Hypotheses

The structural assumption is decomposed into three hypotheses, ordered from weakest to strongest. Each is independently testable.

**H1 — Structure exists.**
Hidden states extracted from π0.5 for LIBERO frames cluster by task identity at a level above chance, measured by silhouette score and k-nearest-neighbor classification accuracy on cosine distance.

**H2 — Structure exceeds the text-only baseline.**
The clusters produced by π0.5 hidden states are *better separated* than clusters produced by CLIP text embeddings of the same instructions in isolation. This justifies caching VLM-side latents rather than caching just CLIP text embeddings, which would be trivial.

**H3 — Structure is retrieval-applicable.**
k-nearest-neighbor classification accuracy on raw (un-projected) hidden states is high enough that retrieval-based latent reuse would predict the correct task class significantly above random.

H1 is necessary but not sufficient. H2 is what justifies the VLM-latent retrieval architecture over a trivial text-keyed alternative. H3 is the most direct retrieval-applicability metric.

## 4. Pre-Registered Success Thresholds

These thresholds are locked before running. Post-hoc adjustment requires explicit written justification appended to this document, with timestamp and reason.

`N_classes` denotes the number of distinct task labels actually present in the sample (will be observed after extraction; LIBERO has 130 task instructions but the random sample may not hit all of them).

| Hypothesis | Metric | Random baseline | Probe success | Strong success |
|---|---|---|---|---|
| H1 | Silhouette score (cosine, raw L2-normalized vectors) | ~0.0 | ≥ 0.15 | ≥ 0.30 |
| H1 | 5-NN classification accuracy (cosine, raw vectors) | 1/N_classes | ≥ 3× random | ≥ 10× random |
| H2 | Δ silhouette: π0.5 prefix − CLIP text | 0 | ≥ +0.05 | ≥ +0.15 |
| H2 | Δ 5-NN accuracy: π0.5 prefix − CLIP text | 0 | ≥ +5 pp | ≥ +15 pp |
| H3 | 5-NN accuracy on raw hidden states (cosine) | 1/N_classes | ≥ 50% | ≥ 80% |

### Decision rule

| Outcome | Decision |
|---|---|
| All three probe-success thresholds met | Proceed to full retrieval implementation on π0.5 LIBERO (Probe 1) AND probe OpenHelix `<ACT>` in parallel |
| H1 + H3 met, H2 fails (CLIP text equally good) | Reformulate retrieval key to use CLIP text only. Architecture simpler, paper motivation weaker but viable |
| H1 fails at probe threshold | Run Probe 0a (alternative pooling, fixed-t suffix) before declaring failure |
| H1 fails across all pooling variants | Run Probe 0c (OpenHelix `<ACT>` token) before abandoning direction |
| H1 fails on both π0.5 and OpenHelix `<ACT>` | Reformulate problem. Candidate paths: action-segment retrieval, hybrid keys, learned retrieval head |

## 5. Experimental Design

### 5.1 Extraction procedure

For each sampled frame, run the training-path fused forward of π0.5 implemented in `prefix_suffix_hidden_states()`:

1. Preprocess image and tokenize instruction
2. Construct action input as $x_t = t \cdot \text{noise} + (1-t) \cdot \text{action}$, with $t \sim \mathcal{U}[0,1]$ and noise freshly sampled
3. Embed prefix (image + text tokens) and suffix (action + time tokens)
4. Build 4D attention mask via `make_att_2d_masks` and position ids via `cumsum(pad_masks) - 1`
5. Run joint `paligemma_with_expert.forward` with `use_cache=False`
6. Return prefix hidden states `[B, S_pref, D]` and last `chunk_size` suffix hidden states `[B, chunk_size, D]`

Both streams are mean-pooled along the sequence dimension to produce one vector per frame per stream. This is the **baseline pooling**; alternative pooling strategies are scoped under Issue 3 (§7).

### 5.2 Data sampling

`N_SAMPLES = 800` frames are sampled uniformly at random with `SEED = 42` from the full LIBERO dataset. For each base frame:

- `NOISE_DRAWS_PER_FRAME = 3` independent forward passes are run with different noise/time draws (changes suffix path each pass, prefix is identical across draws)
- `TEMPORAL_JITTER_FRAMES = 24` random temporal offset is applied within the same episode (uniform ± 24 frames, clamped to episode bounds)

Total expected embeddings per stream: ~2400.

### 5.3 Analysis pipeline

1. **Dimensionality reduction.** StandardScaler → PCA(50) → UMAP(3) with cosine metric, `n_neighbors=15`, `random_state=42`. PCA-only 3D and UMAP 3D produced for both prefix and suffix (4 panels total).

2. **Cluster quality.** Silhouette score on raw L2-normalized vectors with cosine metric (primary). Also compute silhouette in PCA(50) space for sanity check.

3. **k-NN retrieval accuracy.** 5-fold cross-validated `KNeighborsClassifier(n_neighbors=5, metric='cosine')` on raw L2-normalized vectors. Computed separately for prefix and suffix streams.

4. **CLIP text baseline.** Same pipeline (silhouette, kNN, UMAP visualization) on CLIP text embeddings of the same instructions. Encoder: `openai/clip-vit-large-patch14`, text-only path.

5. **Intra-class vs inter-class distance.** For each embedding, compute mean cosine distance to same-label samples vs to different-label samples. Report ratio.

6. **Visualization.** Interactive 3D plotly HTML (for exploration); static matplotlib PNG (for the paper).

### 5.4 Outputs (committed artifacts)

| File | Contents |
|---|---|
| `outputs/probe-0/action_clusters_interactive.html` | Interactive 3D plotly, 4 panels (PCA/UMAP × prefix/suffix) |
| `outputs/probe-0/action_clusters.png` | Static 4-panel figure for paper |
| `outputs/probe-0/clip_text_baseline.html` | CLIP text equivalent visualization |
| `outputs/probe-0/metrics.json` | All silhouette, kNN, intra/inter numbers for prefix / suffix / CLIP text |
| `outputs/probe-0/embeddings.npz` | Raw `X_vlm`, `X_expert`, `X_clip_text`, `labels` for downstream reuse |
| `outputs/probe-0/run_metadata.json` | Seed, dataset commit hash, model checkpoint hash, dtype, wall-clock |

## 6. Pre-Registered Result Interpretation Template

When the probe is complete, fill in the following block in this document (do not edit thresholds in §4):

```
RESULTS (filled after run, date: ____)

Sample size:               N = ____ embeddings, N_classes = ____ task labels

Silhouette (cosine, raw):
  π0.5 prefix:             s = ____
  π0.5 suffix:             s = ____
  CLIP text baseline:      s = ____

5-NN accuracy (cosine, raw):
  π0.5 prefix:             a = ____ (random = ____)
  π0.5 suffix:             a = ____
  CLIP text baseline:      a = ____

Intra/inter cosine ratio:
  π0.5 prefix:             ____
  π0.5 suffix:             ____
  CLIP text baseline:      ____

Hypotheses outcome:
  H1: [met-strong / met / failed]
  H2: [met-strong / met / failed]
  H3: [met-strong / met / failed]

Decision per §4: ____
```

## 7. Known Issues in the Current Implementation

The current extraction script has the following gaps. They must be addressed before the probe is considered conclusive.

### Issue 1 — Labels are long-horizon task instructions, not action primitives

LIBERO task strings span multiple atomic actions (reach, grasp, lift, transfer, release). Frames within one episode share a task label but execute different actions. This dilutes any action-level structure. The probe in its current form tests *task-instruction* clusters, not *action-primitive* clusters. Result interpretation must explicitly reflect this scope.

**Mitigation:** Document explicitly that this probe is at task-instruction granularity. Action-segment-level analysis is a separate follow-up probe (segment episodes by gripper state changes or trajectory inflection points, label each segment).

### Issue 2 — Suffix hidden states are noise-and-time dependent

The flow-matching training path uses $x_t = t \cdot \text{noise} + (1-t) \cdot \text{action}$ with random $t \sim \mathcal{U}[0,1]$ per forward. Suffix hidden states vary with $t$ independently of semantic content. `NOISE_DRAWS_PER_FRAME = 3` averages over only 3 draws, insufficient to cancel this variance. Suffix cluster signal will be artificially weakened.

**Mitigation:** Rerun the suffix analysis with fixed $t = 0$ (clean action conditioning) and $t = 0.05$ (near-final denoising step). Compare to the current random-$t$ result. Report all three.

### Issue 3 — Mean pooling may not be the right summary

For prefix, mean pooling averages 256+ visual tokens with ~20 text tokens. Background-heavy visual content can dominate, washing out instruction semantics. The closest analog of OpenHelix's `<ACT>` token is the last hidden state position, not the mean across all positions.

**Mitigation:** Probe with the following pooling variants in addition to the baseline mean-of-all:

- Last text-token hidden state
- Mean of text-token positions only
- Mean of visual-token positions only

Report a small ablation table comparing all four. Pick the strongest one as primary; mention others in the paper.

### Issue 4 — No CLIP text baseline implemented yet

Without this, H2 cannot be tested at all. **This is the highest-priority gap.**

**Mitigation:** Implement before running. Load `openai/clip-vit-large-patch14`, encode each unique LIBERO instruction string, run the same silhouette / kNN / UMAP pipeline. Hard requirement, no waiver.

### Issue 5 — No kNN retrieval accuracy implemented yet

The most direct retrieval-applicability metric is missing.

**Mitigation:** Implement before running. Use `KNeighborsClassifier(n_neighbors=5, metric='cosine')` with `cross_val_score(cv=5)` on raw L2-normalized embeddings. One number per stream. Hard requirement.

### Issue 6 — Silhouette computed in PCA-50 space with default metric

PCA-50 is a lossy projection and default euclidean does not match what a cosine-based retrieval index would use.

**Mitigation:** Compute silhouette on raw L2-normalized vectors with cosine metric (primary). Keep PCA-50 silhouette as secondary diagnostic.

### Issue 7 — Sampling is unstratified

Task category counts will be unequal; episode position within trajectory is random.

**Mitigation:** For this probe, document the imbalance after extraction and verify no single task dominates the sample (e.g., no task > 10% of samples). For action-level follow-up probes (Issue 1), use stratified sampling at fixed timesteps per episode.

### Issue 8 — In-distribution risk

Model is fine-tuned on LIBERO; probe instructions are LIBERO instructions. Cluster quality may reflect memorization.

**Mitigation:** Out-of-scope for this probe but required follow-up before claiming retrieval generalization. Rerun on the base (non-LIBERO-finetuned) checkpoint, or with paraphrased instructions, in a subsequent probe.

## 8. Known Risks to Interpretation

Beyond implementation issues, several conceptual risks affect what the result *means*.

### Risk 1 — False positive from instruction memorization

π0.5_libero saw these exact instruction strings during fine-tuning. Hidden states may cluster well *because* of memorization, not because of generalizable semantic structure. Mitigated by Issue 8 follow-up.

### Risk 2 — False positive from text-only semantics

Even an untrained LLM produces text-dependent hidden states. Cluster structure on prefix may reflect "the LLM block processed text tokens, which were semantically related," not anything specifically multimodal. Mitigated by Issue 4 (CLIP text baseline) — this is exactly the comparison H2 is designed to test.

### Risk 3 — False negative from architecture mismatch

π0.5's hidden states are not designed to be retrieval-friendly. OpenHelix's `<ACT>` token is. A null result on π0.5 does not foreclose the research direction; it triggers an OpenHelix probe (Probe 0c) before any abandonment decision.

### Risk 4 — False negative from action heterogeneity within task

Frames within one LIBERO task have heterogeneous low-level actions. Task-level clustering may look weak because of within-cluster spread driven by action variation. This is real signal that *task-level* retrieval is hard; it does not rule out *action-segment-level* retrieval.

### Risk 5 — Metric mismatch with use case

Silhouette and global kNN classification measure overall structure. The actual use case is "nearest-neighbor retrieval will produce a useful conditioning latent for the action expert," which is more sensitive to local neighborhood coherence than to global separation. A weak silhouette can coexist with good local retrieval. After kNN accuracy is implemented (Issue 5), the kNN number is the more decision-relevant of the two.

## 9. What This Probe Produces vs Does Not Produce

### Will produce

If positive:

- Motivating figure for the paper introduction (3D UMAP of π0.5 prefix hidden states, colored by task, with visible cluster structure)
- Comparison table: π0.5 prefix vs π0.5 suffix vs CLIP text on silhouette, 5-NN, intra/inter ratio
- Methods-section justification: "Hidden states in dual-system VLAs are semantically structured at a level supporting nearest-neighbor retrieval, motivating our architecture"
- Go decision for the next 4 months

If negative:

- Documentation of which hypothesis failed at which threshold
- Trigger conditions for the decision tree in §4
- Reusable extraction pipeline (this script, refined) for subsequent probes

Regardless:

- Pre-registered methodology baked into the paper
- Reusable embedding dataset (`embeddings.npz`) for any future analysis

### Will NOT produce

This probe explicitly does *not* address:

- Whether *retrieved* latents, when substituted for live VLM latents, produce correct actions in closed-loop execution. That is Probe 1+.
- Whether the structure generalizes out-of-distribution to held-out instructions. Probe 0d.
- Whether retrieval is faster than VLM forward in wall-clock terms with realistic memory bank sizes. Probe 2.
- Whether retrieval works at action-segment granularity. Probe 0e.

These are scoped to subsequent probes within the research program.

## 10. Sequencing of Follow-up Probes

| Probe | Trigger | Description |
|---|---|---|
| Probe 0a | H1 fails on π0.5 baseline | Re-run with alternative pooling (last text token, text-only mean, visual-only mean) and fixed $t \in \{0, 0.05\}$ for suffix |
| Probe 0b | Probe-success on π0.5 | OpenHelix `<ACT>` token cluster probe — confirms architecture-independence of the result |
| Probe 0c | H1 fails on all π0.5 variants | OpenHelix `<ACT>` token, as a different-architecture sanity check |
| Probe 0d | Probe-success on π0.5 | Out-of-distribution generalization — re-run on base (non-LIBERO) checkpoint or with paraphrased instructions |
| Probe 0e | Probe-success on π0.5 | Action-segment granularity — segment episodes by gripper events, label each segment, re-run |
| Probe 1 | Probe-success on π0.5 | Full retrieval-augmented inference implementation: build FAISS index from training latents, swap in retrieved latents at inference, measure closed-loop task success |

## 11. Sign-off Checklist

The probe is **not** complete until all of the following are true:

- [ ] Issues 4 (CLIP text baseline) and 5 (kNN accuracy) are implemented in the extraction script
- [ ] Issue 6 (silhouette on raw L2-normalized vectors with cosine) is implemented
- [ ] Both prefix and suffix streams are evaluated
- [ ] All three hypotheses (H1, H2, H3) have computed metrics against their pre-registered thresholds
- [ ] CLIP text baseline is computed — no waiver permitted
- [ ] At least one alternative pooling strategy (Issue 3) is reported alongside the mean-of-all baseline
- [ ] At least one suffix configuration with fixed $t$ (Issue 2) is reported alongside random-$t$
- [ ] Distribution of task labels in the sample is reported, with verification that no single task exceeds 10% of samples
- [ ] All outputs in §5.4 are committed to the repo with `run_metadata.json` populated
- [ ] Results block in §6 is filled in with observed numbers
- [ ] Decision per §4 is written and dated

## 12. Amendments Log

| Date | Amendment | Justification |
|---|---|---|
| 2026-05-13 | Document locked | Initial pre-registration |

(Any post-lock changes go here. Format: date, what changed, why.)

---

**End of pre-registration.** Any future edits to thresholds in §4 or hypotheses in §3 must be appended in §12 with explicit justification before they can be applied to the results interpretation in §6.
