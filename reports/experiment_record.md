# 实验记录模板：Amazon LLM4Rec（已填入本次运行）

## A. 实验基本信息

- 日期：2026-08-05
- 远端：autodl-721
- 项目：llm4rec-amazon
- 数据：Amazon Reviews 2018 / All_Beauty / 5-core
- 基座：Qwen2.5-1.5B-Instruct
- Semantic ID：RQ-VAE，CPU，feature_dim=256，hidden_dim=128，latent_dim=32，levels=3，codebook_size=16
- GPU：2 × RTX 4090 24 GB
- seed：42
- test protocol：最后一次时间交互留出，151 queries，冻结 test.jsonl，temperature=0

## B. 数据来源与构造记录

### B.1 原始文件

- Review：Amazon Reviews 2018 All_Beauty 5-core gzip JSON
- Metadata：Amazon Reviews 2018 All_Beauty metadata gzip JSON
- 原始字段：reviewerID、asin、unixReviewTime、title、category

### B.2 清洗流水线

1. 流式解析 gzip JSON。
2. 删除 (reviewerID, asin, unixReviewTime) 重复记录。
3. 迭代 user/item 5-core。
4. 按时间排序，使用 asin 做稳定 tie-break。
5. 每个用户最后一次交互作为 test，倒数第二次作为 validation，更早位置作为 train。
6. history 截断到最近 8（RQ-VAE）或 10（ASIN baseline）个商品。
7. 每个 query 采样 1 positive + 15 negatives，负例混合同类别 hard negative 与 popularity-weighted negative。
8. 使用统一 prompt 构造 SFT JSONL、评估 JSONL 和 VERL Parquet。

### B.3 数据快照

| Split | 记录数 | 用途 |
|---|---:|---|
| train | 161 | SID SFT 与 GRPO |
| validation | 151 | 训练期间可选验证/超参 |
| test | 151 | 最终离线评估，仅此处读取 |

- interactions：768
- users：151
- items：28
- candidate size：16
- top_k：10
- semantic raw-path collision groups/items/rate：0 / 0 / 0.0
- codebook utilization（L0/L1/L2）：1.0 / 1.0 / 0.375
- codebook perplexity（L0/L1/L2）：13.2040 / 9.1865 / 3.2881
- collision refresh：请求最多 4 个 seed；seed=42 首次即零碰撞，实际执行 1 次
- fallback：若仍碰撞，使用稳定 ASIN-hash 后缀，不再使用会随插入改号的 X00/X01

### B.4 2026-08-11 全目录召回扩展

- 模型：GRU history encoder + autoregressive 3-level Semantic-ID decoder
- 推理：catalog trie constrained beam search，beam_width=32
- 候选协议：`candidate_mode=full_catalog`，不读取原 JSONL 的 16-candidate list
- catalog：28 个 retained items / 28 个合法 code paths
- split：161 / 151 / 151 queries，validation NDCG@10 选 epoch=20，test 冻结
- test HR@1/5/10/20：0.066225 / 0.092715 / 0.245033 / 0.655629
- test NDCG@10：0.127520
- popularity HR@10 / NDCG@10：0.105960 / 0.045941
- 本机 CPU mean latency：15.9ms/query，仅为原型均值，不是生产 P99
- 结论边界：证明全目录检索链路而非工业规模；All_Beauty 目录太小，需在更大类别复现

## C. Prompt / 输出约束

System：要求只输出严格 JSON。

User：包含用户历史的 item_id/title/category，以及 16 个候选的 item_id/title/category。

Assistant target：

~~~json
{"recommendations":["S030712","S010203", "..."]}
~~~

解析约束：

- 必须是单个 JSON object；
- key 必须是 recommendations；
- list 长度为 10；
- ID 必须全部来自候选集合；
- 不允许重复；
- Semantic ID 由 semantic_ids.json 映射回 ASIN。

## D. SFT 记录

- Config：configs/sft_qlora_rqvae_1p5b.yaml
- Adapter：outputs/sft/qwen2.5-1.5b-all-beauty-rqvae
- Quantization：NF4 + double quantization，BF16 compute
- LoRA：r=8，alpha=16
- gradient checkpointing：on
- batch：1
- steps：6
- train loss：0.7169
- eval loss：0.5751
- 监控：outputs/sft/train_*.log

## E. GRPO 记录

- Config：configs/grpo_verl_rqvae_1p5b_dual.yaml
- Log：outputs/grpo/train_20260805_182438.log
- actor：Qwen2.5-1.5B + SFT LoRA + GRPO LoRA
- reference：shared frozen base / ref_in_actor
- rollout：vLLM，tensor parallel=2，n=4
- group_size：4
- train batch：4
- micro batch per GPU：1
- prompt/response：928/96
- temperature/top_p：0.8/0.95
- KL loss coefficient：0.01
- total updates：40
- checkpoint：global_step_25、global_step_40
- export：global_step_40/export/lora_adapter
- actor peak：约 11.21 GB allocated、13.31 GB reserved / GPU
- step time：约 38–40 秒
- final reward mean（step40）：0.36427
- final rollout KL（step40）：0.00430

### E.1 保存故障与修复

第一次重跑在 step25 遇到 CUDA error: invalid argument。根因是 LoRA 参数引用在 FSDP summon_full_params 退出后仍指向 sharded DTensor，CPU copy 触发 CUDA runtime 错误，并非显存不足。补丁 patches/verl_lora_checkpoint_cpu.patch 改为在上下文内部筛选、拷贝 LoRA 参数。第二次 40-step 运行在 step25 和 step40 保存成功。

## F. Test 评估记录

命令：

~~~bash
bash scripts/evaluate.sh configs/eval_rqvae_1p5b.yaml \
  --adapter-path outputs/sft/qwen2.5-1.5b-all-beauty-rqvae \
  --output-file outputs/evaluation/sft_rqvae_1p5b_predictions.jsonl \
  --metrics-file outputs/evaluation/sft_rqvae_1p5b_metrics.json
~~~

GRPO 只替换 adapter-path 为：

~~~text
outputs/grpo/qwen2.5-1.5b-all-beauty-rqvae-dual/global_step_40/export/lora_adapter
~~~

结果：

| Metric | SFT-only | SFT+GRPO | Delta |
|---|---:|---:|---:|
| Recall@10 | 0.536424 | 0.622517 | +0.086093 |
| HR@10 | 0.536424 | 0.622517 | +0.086093 |
| NDCG@10 | 0.279774 | 0.309018 | +0.029243 |
| format_valid | 0.476821 | 0.582781 | +0.105960 |
| duplicate_free | 0.860927 | 0.953642 | +0.092715 |
| latency/sample | 4.612 | 4.723 | +0.111 |

## G. 错误分析

| Error | SFT-only | SFT+GRPO |
|---|---:|---:|
| duplicate samples | 21 | 7 |
| out-of-candidate | 64 | 56 |
| wrong list length | 1 | 1 |
| hit samples | 81 | 94 |
| mean target rank when hit | 4.988 | 5.287 |

结论：GRPO 同时提升了命中率和格式行为；rank 分布仍有改善空间，后续可提高候选约束和 unknown-item penalty。

## H. 报告中必须披露的限制

1. All_Beauty 是小类别工程数据，不是大规模 Amazon 结论。
2. 只有单一 next-item positive，Recall=HR。
3. test 只用于最终评估，不能据此调参。
4. 40 step/161 train queries 是可复现实验预算，不等价于工业全量训练。
5. DataLoader worker 的清理告警已记录，但没有影响最终 checkpoint/adapter。
6. 后续学术实验应扩展到更大类别、更多候选和 bootstrap 置信区间。

## I. 交付清单核对

- [ ] README.md（数据构造、训练、评估、OOM 处理）
- [ ] requirements.txt / pyproject.toml
- [ ] configs / scripts / src / tests / patches
- [ ] data/processed/all_beauty_rqvae（含 train/val/test、SFT、VERL、semantic_ids）
- [ ] SFT adapter
- [ ] GRPO global_step_40 adapter
- [ ] GRPO 与 SFT 日志
- [ ] test predictions / metrics / comparison / error analysis
- [ ] final_report.md 与本实验记录
- [ ] SHA256 校验文件
