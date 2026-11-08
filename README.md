# LLM4Rec-Amazon: Qwen2.5 + RQ-VAE Semantic ID + VERL GRPO

This project implements a reproducible retrieval-and-reranking generative sequential recommendation pipeline:

1. **RQ-VAE Semantic ID construction** with collision-aware refresh and stable collision suffixes.
2. **Full-catalog hierarchical retrieval** with an autoregressive code decoder and valid-catalog trie beam search.
3. **SID / Sequence Instruction Tuning** with Qwen2.5 and 4-bit QLoRA over retrieved/fixed candidates.
4. **GRPO alignment** with VERL, a vLLM rollout engine, `group_size=4`, a continuous HR/NDCG reward, strict JSON penalties and duplicate penalties.

The original ASIN pipeline remains the default baseline. The RQ-VAE variant is
enabled by `configs/data_rqvae.yaml` and produces compact identifiers such as
`S030712`; these are mapped back to ASINs after generation. This makes the
project support both direct catalog-ID generation and a Semantic-ID generative
recommendation setup without invalidating the original baseline results.

The completed memory-safe dual-4090 experiment uses `Qwen/Qwen2.5-1.5B-Instruct` for both SID and GRPO. The 3B configuration remains available as an engineering baseline, but its VERL/vLLM checkpoint path is close to the 24GB single-card limit and is not the reported full GRPO result.

## Project layout

```text
llm4rec-amazon/
├── configs/                    # data, QLoRA, GRPO and evaluation configs
├── scripts/                    # one-command setup/data/train/eval/demo entry points
├── src/llm4rec/
│   ├── prompts.py              # one prompt contract for SID and GRPO
│   ├── schema.py               # strict JSON parser
│   ├── data/                   # Amazon download, k-core preprocessing, datasets
│   ├── training/               # QLoRA SID, VERL launcher, custom rewards
│   ├── semantic_ids/           # RQ-VAE, collision diagnostics, global retriever
│   ├── evaluation/             # Recall/HR/NDCG and baseline comparison
│   └── inference/              # single-sample demo
├── tests/                      # prompt, metric and reward unit tests
├── patches/                    # pinned VERL compatibility patches
├── data/{raw,processed}/
├── outputs/{global_retrieval,sft,grpo,evaluation}/
└── reports/                    # experiment record and final report templates
```

## Task definition

The project now has two explicit task definitions. Global retrieval receives
only a chronological history and searches every item in `semantic_ids.json`;
it never reads the example's candidate list. The Qwen reranker then ranks a
16-item candidate pool containing the held-out next item. Its input contains:

- chronological user history, capped at the most recent 10 items;
- each history item's `item_id`, title and category;
- candidate item IDs and the same metadata.

The model must return exactly:

```json
{"recommendations":["B000...","B001...","B002..."]}
```

The actual list contains exactly 10 distinct candidate IDs in descending preference order. The compact list schema avoids repeating JSON keys ten times, reducing response tokens and vLLM KV-cache pressure. Raw ASIN generation is not used for retrieval: the lightweight retriever factorizes a Semantic ID path and constrains it with the catalog trie. Qwen remains the expressive but more expensive candidate ranker. Industrial deployment can merge this recall channel with collaborative filtering/two-tower ANN before reranking.

## End-to-end data-to-recommendation trace

The complete path from Amazon's original files to a recommendation is:

```text
Amazon Reviews 5-core gzip                 Amazon metadata gzip
        │                                           │
        └────────────── download_amazon.py ─────────┘
                              │
                 preprocess.py: parse + deduplicate
                              │
              iterative 5-core + timestamp sort + LOO split
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
  ASIN baseline                              RQ-VAE extension
  item_id = ASIN                         title/category → hash features
                                                → RQ-VAE residual codes
                                                → S030712 Semantic ID
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ├── global_retriever.py
                              │   history → p(c1,c2,c3|h)
                              │   catalog-trie beam → full-catalog Top-M
                              │
                              └── candidate sampler + prompts.py
                                  target + hard/popularity negatives
                                  history/candidates → Qwen reranker
                              │
                 sft_train.jsonl / verl_train.parquet
                              │
                Qwen2.5-1.5B NF4 QLoRA SFT adapter
                              │
                    VERL GRPO + vLLM rollout n=4
                              │
                   structured JSON recommendations
                              │
        schema.py validates → candidate map decodes Semantic ID → ASIN
                              │
                 Recall/HR@10 + NDCG@10 + format audit
```

For one raw interaction, the source records look like:

```json
{"reviewerID":"A2...","asin":"B00BSE3III","unixReviewTime":138...}
{"asin":"B00BSE3III","title":"...","category":["Beauty","..."]}
```

After preprocessing, the same next-item example contains a chronological
`history`, a 16-item `candidates` list, `target_item_id` (the original ASIN),
and `target_recommendation_id` (the output identifier). In ASIN mode these two
fields are equal. In RQ-VAE mode a candidate has both `item_id: "S030712"`
and `asin: "B00BSE3III"`, so the model never needs to generate a long opaque
ASIN and evaluation can still report the real product ID.

The exact command sequence is:

```bash
# 1. Download Amazon Reviews/metadata and build the ASIN baseline
bash scripts/prepare_data.sh configs/data.yaml

# 2. Optional Semantic-ID branch (RQ-VAE runs before preprocessing)
bash scripts/prepare_data.sh configs/data_rqvae.yaml

# 3. Train the full-catalog hierarchical Semantic-ID retriever
bash scripts/train_global_retriever.sh configs/global_retriever.yaml

# 4. Train Qwen with 4-bit QLoRA
bash scripts/train_sft.sh configs/sft_qlora.yaml
# or: bash scripts/train_sft.sh configs/sft_qlora_rqvae.yaml

# 5. Align the SFT policy with VERL GRPO/vLLM
bash scripts/train_grpo.sh configs/grpo_verl.yaml
# or: bash scripts/train_grpo.sh configs/grpo_verl_rqvae.yaml

# 6. Greedy offline evaluation and one-user demo
bash scripts/evaluate.sh configs/eval.yaml --adapter-path outputs/sft/qwen2.5-3b-all-beauty
bash scripts/run_demo.sh --config configs/eval.yaml --sample-index 7
```

The split contract is strict: `sft_train.jsonl` and `verl_train.parquet` are built only from earlier chronological positions; `sft_val.jsonl`/`verl_val.parquet` are validation data; `test.jsonl` is the final frozen leave-one-out test set. The test file is never read by SFT or GRPO and is used only by `llm4rec.evaluation.evaluate` after adapters are exported.

The run artifacts are intentionally stage-specific. `outputs/prepare_data.log`
records raw parsing and split statistics; `outputs/rqvae_train_*.log` records
codebook/reconstruction loss and code utilization; `outputs/sft/train_*.log`
records QLoRA loss and peak VRAM; `outputs/grpo/train_*.log` records rollout
reward/KL/step time; and `outputs/evaluation/*predictions.jsonl` preserves the
raw Qwen JSON, parser errors, decoded ASINs and per-sample metrics. A convenient
pre-GRPO guard is:

```bash
python -m llm4rec.data.inspect_prompt_lengths \
  --file data/processed/all_beauty_rqvae/train.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct
```

This reports min/max/mean/p95 chat-template token counts. Keep the maximum below
`data.max_prompt_length`; otherwise VERL correctly filters the record rather than
silently truncating away candidates.

## Single-4090 memory plan

Approximate peak usage depends on allocator fragmentation and the installed vLLM version:

| Stage | 3B configuration | Estimated peak | Measured allocated / reserved |
|---|---|---:|---:|
| SID | NF4 QLoRA, batch 1, seq 1024, grad accumulation 16 | 8–12GB | 5.36 / 5.96GB |
| Offline eval | NF4 base + LoRA, batch 1 | 4–7GB | 2.20 / 2.29GB |
| GRPO | BF16 LoRA actor + vLLM, group 4, 896+128 tokens, offload | 19–23GB | 17.75 / 22.00GB |
| GRPO dual-card Semantic-ID | 1.5B LoRA actor + vLLM TP=2, group 4, 928+96 tokens | 12–15GB/card | 11.21 allocated / 13.31 reserved per card |

The GRPO margin is small. Defaults therefore enable gradient checkpointing, optimizer/reference offload, dynamic token batches, vLLM cache release, eager mode and `gpu_memory_utilization=0.30`. The 3B vLLM server needs this small reservation increase because 0.28 leaves only 0.02GB for KV cache after loading weights, below the 1024-token minimum. Actor parameter offload is disabled because single-GPU FSDP selects `NO_SHARD`, which is incompatible with VERL's LoRA weight summon when `offload_to_cpu=True`. Remove-padding and FlashAttention are disabled on the inspected host because its system CUDA toolkit (12.8) does not match the CUDA runtime bundled with the current vLLM PyTorch wheel (13.0); the GRPO actor explicitly uses PyTorch SDPA instead.

The RQ-VAE stage is intentionally CPU-only by default. It uses a 256-dimensional
hashing text feature, a 128-unit MLP, a 32-dimensional latent, three residual
codebooks of size 16 and a maximum batch of 128. It adds effectively zero GPU
memory to SFT/GRPO. Semantic-ID GRPO is more conservative than the formal ASIN
run: prompt/response limits are 928/96, vLLM `max_num_seqs` is 4 and
`gpu_memory_utilization` is 0.30. The earlier 0.25 smoke had no room for even
one vLLM KV-cache block after the 3B weights loaded, so lowering this particular
reservation is counterproductive; control peak use with `max_num_seqs=4` and
bounded token lengths instead.

### Reference-model implementation note

Current VERL LoRA training computes reference log-probabilities by disabling the adapter on the actor and reusing the frozen base weights (`ref_in_actor`). It does **not** support a separate bitsandbytes 4-bit FSDP reference. This shared reference consumes no duplicate base-model allocation and is more memory-efficient than a second 4-bit model while being mathematically the intended pre-RL reference policy. The SFT and offline audit paths do load the base in NF4. The launcher rejects unsupported reference modes instead of silently pretending that an independent 4-bit VERL reference was loaded.

## Completed formal result

The completed reported run is the Qwen2.5-1.5B Semantic-ID experiment on two RTX 4090s. The frozen test set has 151 queries:

| Metric | SFT-only | SFT+GRPO | Delta |
|---|---:|---:|---:|
| Recall@10 / HR@10 | 0.536424 | 0.622517 | +0.086093 |
| NDCG@10 | 0.279774 | 0.309018 | +0.029243 |
| Strict format valid | 0.476821 | 0.582781 | +0.105960 |
| Duplicate-free | 0.860927 | 0.953642 | +0.092715 |

The step40 GRPO run used `outputs/grpo/train_20260805_182438.log` and exported `global_step_40/export/lora_adapter`. SFT-only and SFT+GRPO were evaluated on the exact same `data/processed/all_beauty_rqvae/test.jsonl` with greedy decoding. The corresponding metrics, predictions, comparison table and error-analysis JSON files are under `outputs/evaluation/`. The older 3B one-card result is retained in archived logs and should be described as an earlier baseline, not mixed with this 1.5B comparison.

GRPO improved hit-based ranking metrics and clearly reduced repeated recommendations in this 1.5B run. Error analysis still found a worse mean target rank among hits (4.988 to 5.287), while out-of-candidate errors decreased from 64 to 56 but remained the dominant format issue. See `reports/final_report.md` and `outputs/evaluation/comparison_rqvae_1p5b.md` for the full interpretation; the limitation is preserved rather than selecting a checkpoint on the test set.

## Environment setup

The tested remote layout uses the data disk for the Conda environment and caches:

```bash
cd /root/autodl-tmp/llm4rec-amazon
bash scripts/setup_env.sh
```

The script creates `/root/autodl-tmp/envs/llm4rec`, installs vLLM first, pins the inspected VERL commit, conditionally builds FlashAttention only when CUDA ABIs match, installs a pure-PyTorch padding-helper fallback when they do not match, and installs this package editable. Override paths if needed:

```bash
LLM4REC_ENV_PREFIX=/path/to/env VERL_SRC=/path/to/verl bash scripts/setup_env.sh
```

On the inspected AutoDL network, Hugging Face's primary endpoint timed out while `hf-mirror.com` was reachable; `scripts/common_env.sh` therefore defaults `HF_ENDPOINT` to the mirror and sets `HF_HUB_DISABLE_XET=1` because mirror downloads through Xet CAS returned 401. Set `HF_ENDPOINT=https://huggingface.co` and opt back into Xet only if the primary endpoint is available in your region. The inspected VERL commit requires `5.5.3 <= transformers < 5.11` (excluding 5.6.0), while vLLM 0.24 requires Transformers 5.x; do not downgrade it to a 4.x wheel.

## Data preparation

The default category is Amazon 2018 `All_Beauty` 5-core:

```bash
bash scripts/prepare_data.sh configs/data.yaml
```

Processing steps:

1. stream review and metadata gzip files;
2. normalize `reviewerID`, `asin`, `unixReviewTime`;
3. iteratively enforce 5-core after deduplication;
4. sort each user sequence by timestamp;
5. use leave-two-out validation/test and up to three earlier train positions;
6. keep the latest 10 history items;
7. sample half same-category hard negatives and half popularity-weighted negatives;
8. emit SID JSONL, evaluation JSONL and VERL Parquet from the same prompt builder.

Main outputs:

```text
data/processed/all_beauty/
├── stats.json
├── sft_train.jsonl
├── sft_val.jsonl
├── verl_train.parquet
├── verl_val.parquet
├── val.jsonl
└── test.jsonl
```

## RQ-VAE Semantic ID extension

RQ-VAE is a separate item-encoding stage, not a second language model. It
streams the same Amazon metadata, builds deterministic CPU hashing features from
title/category text, trains a small residual vector quantizer, and exports:

```text
data/processed/all_beauty_rqvae/
├── semantic_ids.json       # ASIN <-> compact Semantic ID mapping
├── rqvae.pt                # tiny RQ-VAE checkpoint
├── rqvae_stats.json
└── sft_*/verl_*/test.jsonl # prompts now contain Semantic IDs
```

Run the extension end to end:

```bash
bash scripts/prepare_data.sh configs/data_rqvae.yaml
```

This automatically runs `llm4rec.semantic_ids.train_rqvae` before preprocessing.
To train only the encoder:

```bash
bash scripts/train_rqvae.sh configs/rqvae.yaml
```

Semantic IDs use the fixed-width compact form `S030712`. Training now records
per-level utilization/perplexity and raw path-collision metrics. After neural
training, residual k-means refresh starts from seed 42; additional deterministic
seeds are tried only when a raw collision is detected, and the assignment with
the fewest collisions (then the lowest reconstruction loss) is retained. If a
collision still remains, each colliding item receives a stable item-hash suffix
such as `X9A21F4C0`, so insertion of a new item does not renumber older IDs as an
ordinal `X00/X01` scheme would. The GRPO ground truth includes a small optional
hierarchical-prefix reward (`prefix_reward_weight=0.05`) and a stronger
out-of-candidate penalty (`unknown_item_penalty=0.20`). Exact HR/NDCG remains the
primary metric. Offline evaluation reports both generated Semantic IDs and their
decoded ASINs.

The 2026-08-11 CPU retrain kept all 28 public base IDs collision-free:

| Diagnostic | Value |
|---|---:|
| unique base paths / items | 28 / 28 |
| collision groups / extra collision items | 0 / 0 |
| codebook utilization by level | 100%, 100%, 37.5% |
| codebook perplexity by level | 13.204, 9.186, 3.288 |
| refreshed reconstruction loss | 0.0028069 |

## Full-catalog Semantic-ID retrieval

The Qwen SFT/GRPO branch remains a candidate reranker. A separate lightweight
retrieval stage now removes the former requirement that the true item be placed
inside a 16-item prompt. `SemanticSequenceRetriever` encodes the user history
once with a GRU and factorizes the next-item path as

```text
p(c1, c2, c3 | history)
= p(c1 | history) p(c2 | history,c1) p(c3 | history,c1,c2)
```

At inference, `SemanticCatalog` exposes only prefixes that lead to a real item.
Trie-constrained beam search therefore cannot emit an out-of-catalog base code.
Raw-path collision buckets are expanded at the leaf and use a train-only
popularity prior solely as a deterministic tie-break. Train and evaluate with:

```bash
bash scripts/train_global_retriever.sh configs/global_retriever.yaml
```

The retained All_Beauty experiment ignores every record's 16-item candidate
list and retrieves from all 28 mapped items. Frozen-test results are:

| Full-catalog metric | Semantic retriever | Popularity baseline |
|---|---:|---:|
| HR / Recall@1 | 0.0662 | 0.0066 |
| HR / Recall@5 | 0.0927 | 0.0265 |
| HR / Recall@10 | 0.2450 | 0.1060 |
| NDCG@10 | 0.1275 | 0.0459 |
| HR / Recall@20 | 0.6556 | 0.6689 |

The measured CPU prototype latency was 15.9 ms/query on the inspected Windows
host, but this is not a production P99 claim. The catalog has only 28 retained
items; industrial scaling still requires a much larger category, sharded trie
or hierarchical serving, multi-seed confidence intervals, online freshness,
and an ANN/fallback path. The intended two-stage system is global Semantic-ID
retrieval followed by the existing Qwen+GRPO candidate reranker.

## Stage 1: SID QLoRA

Formal run:

```bash
bash scripts/train_sft.sh configs/sft_qlora.yaml
```

Two-step smoke run:

```bash
bash scripts/train_sft.sh configs/sft_smoke.yaml
```

Only assistant JSON tokens contribute to the loss. Prompt tokens are labeled `-100`. If a sample exceeds the limit, the truncator keeps the system instruction plus the newest history/candidate region and always preserves the full target JSON.

Important SID knobs:

- `max_length`: lower from 1024 to 768 first on OOM.
- `gradient_accumulation_steps`: increases effective batch without increasing peak VRAM.
- `lora_rank`: 16 is the formal default; 8 saves memory with a small capacity tradeoff.
- `lora_target_modules`: all Qwen attention and MLP projections are adapted.
- `learning_rate`: start at `2e-4`; try `1e-4` if format learning is stable but ranking overfits.
- `flash_attention_2`: optional for SID; disabled by default for portability.

Semantic-ID SFT uses the same implementation with a separate output directory:

```bash
bash scripts/train_sft.sh configs/sft_qlora_rqvae_1p5b.yaml
```

## Stage 2: VERL GRPO

First validate the exact Hydra command without allocating a model:

```bash
python -m llm4rec.training.train_grpo_verl --config configs/grpo_verl.yaml --dry-run
```

Formal run:

```bash
bash scripts/train_grpo.sh configs/grpo_verl.yaml
```

Semantic-ID run (after the RQ-VAE data and Semantic-ID SFT adapter exist):

```bash
python -m llm4rec.training.train_grpo_verl \
  --config configs/grpo_verl_rqvae.yaml --dry-run
bash scripts/train_grpo.sh configs/grpo_verl_rqvae.yaml
```

Two RTX 4090s are the recommended configuration for Semantic-ID GRPO because
FSDP shards the actor while vLLM tensor-parallelizes rollout weights across both
cards. The hard `group_size=4` remains unchanged:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train_grpo.sh configs/grpo_verl_rqvae_1p5b_dual.yaml
```

The dual-card config sets `n_gpus_per_node=2` and
`tensor_model_parallel_size=2`. Run `configs/grpo_verl_rqvae_dual_smoke.yaml`
first when moving to new hardware; it performs one update and a real checkpoint
save.

The key configuration is:

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.n=4
actor_rollout_ref.model.lora_adapter_path=<SID adapter>
actor_rollout_ref.actor.use_kl_loss=True
reward.custom_reward_function.path=<rewards.py>
```

The inspected VERL 0.9 development checkout has V1 enabled by default, but its
`transfer_queue` experimental package is not included in the source checkout. The
launcher explicitly sets `trainer.use_v1=false` and uses the still-supported
legacy PPO trainer, which shares the same FSDP actor, vLLM rollout, LoRA reference
and GRPO implementation. If your VERL install includes `transfer_queue`, remove
that override to use V1.

Reward for a parsed list is:

```text
0.45 * HR@10 + 0.45 * NDCG@10 + format_bonus
- wrong-length cost - duplicate cost - out-of-candidate cost
```

For Semantic IDs, an additional bounded prefix reward can provide a weak signal
when the model emits a code sharing early RQ-VAE levels with the target. It is
disabled for the ASIN baseline and never replaces exact target HR/NDCG.

Malformed non-JSON output receives `-1.0`. Valid output with the target at rank 1 receives `1.0`. This gives continuous rank-sensitive feedback while preserving a decisive format signal.

Important GRPO knobs:

- `group_size=4`: fixed by the experiment design. In this VERL version `ppo_mini_batch_size` is measured in source prompts and must divide `train_batch_size`; VERL multiplies it by `group_size` internally.
- `kl_loss_coef=0.01`: safe initial value. Try `0.005` if policy improvement is too weak; try `0.02`–`0.05` if KL or format collapse grows.
- `rollout_gpu_memory_utilization=0.30`: lower to `0.22` on KV-cache OOM; the Semantic-ID config starts at `0.25`.
- `max_prompt_length=896`, `max_response_length=128`: the JSON normally fits in 100–120 tokens.
- `temperature=0.8`: creates within-group diversity. Too low yields identical rewards and near-zero GRPO advantages.
- `actor_learning_rate=5e-6`: try `2e-6` if KL spikes or JSON validity drops.
- `entropy_coeff=0`: GRPO sampling already supplies exploration; a positive value can destabilize strict JSON.

### Exporting the GRPO LoRA adapter

VERL checkpoints are sharded. Export the latest step:

```bash
bash scripts/export_grpo_adapter.sh \
  outputs/grpo/qwen2.5-1.5b-all-beauty-rqvae-dual global_step_40 \
  outputs/sft/qwen2.5-1.5b-all-beauty-rqvae
```

The evaluation adapter is then:

```text
outputs/grpo/qwen2.5-1.5b-all-beauty-rqvae-dual/global_step_40/export/lora_adapter
```

## Offline evaluation and baseline comparison

SFT-only:

```bash
bash scripts/evaluate.sh configs/eval_rqvae_1p5b.yaml \
  --adapter-path outputs/sft/qwen2.5-1.5b-all-beauty-rqvae \
  --output-file outputs/evaluation/sft_rqvae_1p5b_predictions.jsonl \
  --metrics-file outputs/evaluation/sft_rqvae_1p5b_metrics.json
```

SFT+GRPO:

```bash
bash scripts/evaluate.sh configs/eval_rqvae_1p5b.yaml \
  --adapter-path outputs/grpo/qwen2.5-1.5b-all-beauty-rqvae-dual/global_step_40/export/lora_adapter \
  --output-file outputs/evaluation/grpo_rqvae_1p5b_predictions.jsonl \
  --metrics-file outputs/evaluation/grpo_rqvae_1p5b_metrics.json
```

Comparison table:

```bash
python -m llm4rec.evaluation.compare \
  --sft outputs/evaluation/sft_metrics.json \
  --grpo outputs/evaluation/grpo_metrics.json \
  --output outputs/evaluation/comparison.md
```

Besides Recall@10 and NDCG@10, evaluation records exact format validity, duplicate-free rate, mean latency and peak allocated VRAM. With a single held-out next item, Recall@10 equals HR@10; both are reported for convention compatibility.

## Inference demo

```bash
bash scripts/run_demo.sh \
  --config configs/eval.yaml \
  --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
  --adapter-path outputs/grpo/qwen2.5-3b-all-beauty/global_step_<N>/export/lora_adapter \
  --sample-index 0
```

For the completed 1.5B smoke adapter, replace the model and adapter paths together. A LoRA adapter must always be loaded on the same base-model size from which it was trained.

## Common failures

### CUDA OOM during vLLM startup

Apply these in order:

1. set `rollout_gpu_memory_utilization: 0.22`;
2. set `max_prompt_length: 768` and `max_response_length: 96`;
3. set `rollout_max_num_seqs: 4`;
4. keep optimizer/reference offload enabled; keep actor parameter offload **false**
   on the inspected one-GPU FSDP setup because `NO_SHARD` cannot summon LoRA
   weights with CPU offload;
5. stop stale Ray/vLLM processes and confirm with `nvidia-smi`;
6. use the 1.5B smoke config to separate code errors from capacity errors.

Do not reduce `group_size`; it is a hard experimental setting. Reduce prompts per global step instead.

### Semantic-ID or RQ-VAE OOM

RQ-VAE itself should run on CPU. Set `device: cpu`, `batch_size: 32` (the code
caps it at 128), `hidden_dim: 128` and `feature_dim: 256`. Do not load Qwen or
vLLM in the same process as RQ-VAE training. If a user intentionally enables a
CUDA RQ-VAE device, use `batch_size: 16` and run it as a separate process before
SFT. For Semantic-ID GRPO, lower `rollout_gpu_memory_utilization` to `0.20`,
`rollout_max_num_seqs` to 2, `max_response_length` to 80 and
`max_token_len_per_gpu` to 1024 before changing model size. The compact
`Sxxxxxx` format is specifically used to keep generated responses short.

### KV cache allocation failure

This is usually vLLM reserve pressure, not PyTorch training activations. Lower `gpu_memory_utilization`, response length and `max_num_seqs`. Keep `free_cache_engine=true`, `enforce_eager=true`, prefix caching off and chunked prefill off for this small irregular workload.

### Overlong prompts filtered by VERL

Inspect titles and candidate size. The data defaults truncate titles to 72 characters and history to 10. Do not change `data.truncation=error` to silent right truncation: right truncation can remove candidates and invalidate the reward contract. Regenerate data with fewer candidates/history items instead.

### FlashAttention build/import error

Confirm `CUDA_HOME=/usr/local/cuda`, `nvcc --version`, and `torch.version.cuda`. Build FlashAttention only when the toolkit and PyTorch CUDA versions match exactly; otherwise keep `memory.remove_padding=false`. Never reuse a FlashAttention wheel compiled for a different PyTorch/CUDA ABI.

### bitsandbytes CUDA setup error

Run `python -m bitsandbytes` in the active environment. Confirm `torch.cuda.is_available()` and avoid installing a CPU-only PyTorch wheel after vLLM. Re-run setup in a clean environment if pip replaced torch.

### LoRA rank/shape mismatch in VERL

The GRPO `lora_rank` and `lora_alpha` must match `adapter_config.json` from SID. Smoke uses rank 8; formal training uses rank 16. Do not pair the smoke adapter with the formal GRPO config.

### GRPO reward variance is zero

Check that `temperature` is not zero, `group_size=4`, and sampled outputs differ. Also inspect format validity: four identical malformed responses all receive `-1`, producing zero relative advantage. More SID format training is preferable to increasing RL learning rate.

### KL grows rapidly or recommendation quality collapses

Lower actor LR, increase `kl_loss_coef`, and verify the SFT adapter is loaded. Typical initial sweep: coefficient `{0.005, 0.01, 0.02}`. Track policy KL jointly with reward; reward alone can improve by exploiting format/ranking shortcuts.

### Repeated recommendations

The prompt requires distinct IDs, the parser marks duplicates invalid, and the reward subtracts `0.15` per repeated entry (capped). If repetition remains high, increase SID examples/epochs before making the RL penalty larger.

## Reproducibility checklist

- Save `stats.json`, resolved YAML configs, Git/VERL commit and `pip freeze`.
- Report the exact sample cap; a smoke result is not a formal baseline.
- Evaluate SFT and GRPO on the same frozen `test.jsonl` with greedy decoding.
- Report format-valid rate next to ranking metrics.
- Record peak GPU memory from both training logs and `nvidia-smi`.
- Run `pytest -q` before training and after environment changes.
