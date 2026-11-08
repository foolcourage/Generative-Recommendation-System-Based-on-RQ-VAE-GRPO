# Amazon LLM4Rec 生成式推荐技术报告

## 1. 摘要

本项目从 Amazon Reviews 2018 的 All_Beauty 5-core 原始 Review 与商品 Metadata 出发，构建带商品标题/类别信息的时间序列推荐数据，并增加带碰撞诊断的 RQ-VAE Semantic ID。系统已形成两阶段：轻量 GRU retriever 按层生成 Semantic-ID code，并通过合法目录 trie beam 做 full-catalog recall；Qwen2.5-1.5B-Instruct 在召回后做候选内重排，使用 QLoRA SFT 与 VERL GRPO 对齐。GRPO 使用 vLLM rollout、group_size=4、LoRA actor、共享冻结 reference、KL 正则和结构化 JSON 奖励。

最终离线评估严格使用从用户时间序列末端留出的 151 条 test.jsonl。全目录召回完全忽略记录中的 16 candidates，在 28-item retained catalog 上取得 HR@10=0.2450、NDCG@10=0.1275，高于 popularity 的 0.1060/0.0459。候选重排的 SFT+GRPO 相比 SFT-only 在 test 上 Recall@10 提升 8.61 个百分点，NDCG@10 提升 2.92 个百分点，JSON 格式有效率提升 10.60 个百分点。两类指标协议不同，不能直接比较。

必须说明：All_Beauty 经过 5-core 后只有 151 个用户、28 个商品，是小规模工程闭环实验，不足以支持 Amazon 大规模学术结论。后续应在 Video_Games、Movies_and_TV 或 Books 上复现实验并报告置信区间。

## 2. 实验对象与硬件

| 项目 | 配置 |
|---|---|
| 原始数据 | Amazon Reviews 2018 All_Beauty 5-core + Metadata |
| 基座模型 | Qwen/Qwen2.5-1.5B-Instruct |
| Semantic ID | CPU RQ-VAE，3 个 residual codebook，每级 16 个 code |
| 全目录召回 | GRU history encoder + autoregressive code decoder + catalog trie beam=32 |
| SFT | NF4 4-bit QLoRA，LoRA rank 8，alpha 16 |
| GRPO | VERL legacy trainer，vLLM rollout，n=4 |
| GPU | 2 × RTX 4090 24 GB |
| 训练步数 | 40 GRPO updates，1 epoch |
| 候选数 / 输出 | 召回：全 28-item catalog；重排：每题 16 候选输出 Top-10 |
| 随机种子 | 42 |
| 软件 | PyTorch 2.11+cu130，Transformers 5.10.4，PEFT 0.20，vLLM 0.24，VERL 0.9 dev |

## 3. 原始 Amazon 数据如何变成推荐样本

### 3.1 原始字段

Review gzip 每行核心字段为：

~~~json
{"reviewerID":"A2...","asin":"B00BSE3III","unixReviewTime":138...}
~~~

Metadata gzip 每行包含：

~~~json
{"asin":"B00BSE3III","title":"product title","category":["Beauty","..."]}
~~~

代码入口是 src/llm4rec/data/download_amazon.py 和 src/llm4rec/data/preprocess.py。

### 3.2 清洗、去重与 5-core

1. 流式读取 gzip，避免把原始文件一次性加载到内存。
2. 统一 reviewerID、asin、unixReviewTime 类型；缺失标题使用空字符串，缺失类别使用 Unknown。
3. 删除 (user_id, item_id, timestamp) 完全重复行。
4. 迭代执行 user/item 5-core，直到稳定。
5. 对每个用户按 (unixReviewTime, asin) 稳定排序，得到时间序列。
6. 历史最多保留最近 10 个 ASIN（Semantic-ID 分支最近 8 个），标题截断到 72 字符，防止 prompt 超长。

本次小规模数据统计：

| 统计量 | 数值 |
|---|---:|
| interactions | 768 |
| users | 151 |
| items | 28 |
| train queries | 161 |
| validation queries | 151 |
| test queries | 151 |
| candidate size | 16 |

### 3.3 时间隔离与 leave-one-out

对每个用户的有序序列，最后一次交互作为 test target，倒数第二次交互作为 validation target，更早位置构造 train query。训练样本目标不会出现在该样本历史中；test 的目标和候选只在 test record 中出现。候选负例从训练可见商品池按类别和流行度采样，避免未来交互泄漏。

因此：

~~~text
SFT/GRPO: train split only
超参检查: validation split only
最终 Recall/NDCG: frozen test split only
~~~

### 3.4 候选构造

每个 query 构造 16 个候选：1 个真实下一商品 + 同类别 hard negatives + popularity-weighted negatives。候选保留 item_id、原始 asin、title、category。Semantic-ID 分支将生成目标从 ASIN 换成紧凑的 S030712，评估时再通过 semantic_ids.json 解码回 ASIN。

2026-08-11 重新训练后，28 个商品仍得到 28 个不同的基础 code。训练器现在显式输出 raw-path collision group/item/rate、各层 utilization 与 perplexity；只有检测到碰撞才继续尝试额外 residual-k-means seed。若多起点 refresh 后仍碰撞，编码器追加仅依赖原始 ASIN 的稳定哈希后缀（如 `X9A21F4C0`）保证可逆，避免新增商品导致旧的 X00/X01 顺序后缀整体改号。

## 4. Prompt 与数据格式

SID 和 GRPO 共用 src/llm4rec/prompts.py 的统一模板。system message 强制模型只输出：

~~~json
{"recommendations":["<candidate_id>","..."]}
~~~

user message 包含用户历史、每个历史商品的 title/category、16 个候选及候选 metadata，并要求恰好 10 个不同 ID、只能使用候选 ID、按偏好降序排列、禁止 Markdown 和解释文本。统一模板保证 SFT 学到的 JSON 行为与 GRPO rollout、离线评估一致。

### 4.1 SFT JSONL

~~~json
{
  "sample_id":"train:A2...:3",
  "messages":[
    {"role":"system","content":"...strict JSON..."},
    {"role":"user","content":"history + candidates + metadata"},
    {"role":"assistant","content":"{\"recommendations\":[\"S030712\", \"...\"]}"}
  ]
}
~~~

SFT 使用 completion-only loss，system/user token 的 label 为 -100，只对 assistant JSON 计算交叉熵。

### 4.2 GRPO Parquet

~~~json
{
  "data_source":"amazon_llm4rec",
  "prompt":[{"role":"system","content":"..."},{"role":"user","content":"..."}],
  "reward_model":{"style":"rule","ground_truth":"{...target_item_id...}"},
  "extra_info":{
    "sample_id":"train:A2...:3",
    "split":"train",
    "target_item_id":"S030712",
    "candidate_ids":["S030712","S010203","..."],
    "identifier_mode":"semantic_id"
  }
}
~~~

GRPO 没有重新从 test 构造 prompt，也没有把 test target 放入训练 parquet。

## 5. RQ-VAE Semantic ID

RQ-VAE 只在 CPU 进程中运行，输入是 title/category 的确定性 hashing TF-IDF-like 256 维特征；网络为 256→128 MLP、32 维 latent，三层 residual quantizer，每层 codebook size=16，batch 上限 128。神经训练后进行 residual k-means codebook refresh，解决小目录上的 codebook collapse：

~~~text
num_items=28
loss=0.0037981
refreshed_loss=0.0028270
unique_base_codes=28
collision_groups=0
collision_items=0
collision_rate=0.0
used_codes_per_level=[16,16,6]
codebook_utilization_per_level=[1.0,1.0,0.375]
codebook_perplexity_per_level=[13.2040,9.1865,3.2881]
collision_refresh_attempts_requested=4
collision_refresh_attempts_executed=1
~~~

RQ-VAE 不与 Qwen/vLLM 同进程，因此不会额外占用 GRPO GPU 显存。

### 5.1 全目录 Semantic-ID 召回

原来的 Qwen+GRPO 任务仍是 16 候选内重排。为补上工业推荐漏斗的召回阶段，项目新增 `SemanticSequenceRetriever`：GRU 只编码一次用户历史，自回归预测三层 residual code，目标分解为

$$
p(c_1,c_2,c_3\mid h)=p(c_1\mid h)p(c_2\mid h,c_1)p(c_3\mid h,c_1,c_2)。
$$

推理使用 `SemanticCatalog` 的合法前缀树约束 beam search：每一级只扩展最终能落到真实目录商品的 code，因而不会生成不存在的基础路径；若叶节点包含碰撞桶，则展开桶内商品并仅用 train-only popularity 作稳定 tie-break。训练/验证/测试直接读取 history 与 held-out target，明确忽略 JSONL 中预构造的 16 个 candidates。

运行命令：

~~~bash
bash scripts/train_global_retriever.sh configs/global_retriever.yaml
~~~

协议：All_Beauty retained catalog 全部 28 items，161/151/151 train/validation/test queries，seed=42，beam width=32，validation NDCG@10 选 epoch 20，frozen test 只评估一次。

| Full-catalog test metric | Semantic retriever | Popularity |
|---|---:|---:|
| HR / Recall@1 | 0.066225 | 0.006623 |
| HR / Recall@5 | 0.092715 | 0.026490 |
| HR / Recall@10 | 0.245033 | 0.105960 |
| NDCG@10 | 0.127520 | 0.045941 |
| HR / Recall@20 | 0.655629 | 0.668874 |

该结果证明“无预置 16 候选的完整目录检索链路”已经跑通，并在 Top-10 明显超过 train-popularity baseline；但 28-item catalog 不是工业规模，且 validation target 高度集中、test 分布更分散，不能用很高的 validation 指标替代 test。当前 Windows CPU 的平均检索时间约 15.9ms/query 只用于本机复现，不是生产 P99。百万目录仍需大类别数据、分片 trie/层级服务、增量索引、跨 seed CI、ANN fallback 与线上 A/B。

## 6. 训练过程

### 6.1 阶段 1：SID QLoRA

使用 Qwen2.5-1.5B-Instruct 的 NF4 4-bit base，计算 dtype 为 BF16，LoRA rank=8、alpha=16，gradient checkpointing 开启，batch=1，短序列截断保留 system 指令和最近候选区：

~~~text
train_loss=0.7169
eval_loss=0.5751
steps=6
~~~

SFT adapter 位于 outputs/sft/qwen2.5-1.5b-all-beauty-rqvae。

### 6.2 阶段 2：VERL GRPO

训练命令：

~~~bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_grpo.sh configs/grpo_verl_rqvae_1p5b_dual.yaml
~~~

核心设置：

~~~text
group_size / rollout.n = 4
tensor_model_parallel_size = 2
train_batch_size = 4
ppo_micro_batch_size_per_gpu = 1
max_prompt_length = 928
max_response_length = 96
vLLM gpu_memory_utilization = 0.30
KL coefficient = 0.01
actor LoRA rank = 8
~~~

奖励函数 src/llm4rec/training/rewards.py：

~~~text
R = 0.45 * HR@10 + 0.45 * NDCG@10
    + format/semantic-prefix bonus
    - wrong_length_penalty
    - duplicate_penalty
    - unknown_item_penalty
~~~

格式错误得到强负奖励；重复推荐和非候选 ID 会被扣分。GRPO 对每个 prompt 采样 4 个回答，组内归一化 advantage，再用 KL 正则限制偏离 SFT policy。

正式日志为 outputs/grpo/train_20260805_182438.log，共 40 step。step25 和 step40 checkpoint 均成功保存；actor 峰值约 11.21 GB allocated、13.31 GB reserved/卡。训练清理阶段出现一次 DataLoader worker signal: Killed，但 step40 指标、checkpoint 与导出 adapter 均完整，未出现 CUDA OOM。

## 7. 严格 test 评估结果

评估使用 test.jsonl 的 151 条记录，temperature=0，Top-10，严格 JSON/candidate parser。单一 next-item 设定下 Recall@10 等于 HR@10，但两者都保留。

| Metric | SFT-only | SFT+GRPO | Delta |
|---|---:|---:|---:|
| Recall@10 | 0.536424 | 0.622517 | +0.086093 |
| HR@10 | 0.536424 | 0.622517 | +0.086093 |
| NDCG@10 | 0.279774 | 0.309018 | +0.029243 |
| format_valid | 0.476821 | 0.582781 | +0.105960 |
| duplicate_free | 0.860927 | 0.953642 | +0.092715 |
| latency/sample | 4.612s | 4.723s | +0.111s |
| peak eval allocated | 1.242GB | 1.242GB | 0 |

错误分析：

| 行为 | SFT-only | SFT+GRPO |
|---|---:|---:|
| hit samples | 81/151 | 94/151 |
| duplicate samples | 21 | 7 |
| out-of-candidate errors | 64 | 56 |
| wrong-list-length errors | 1 | 1 |
| mean target rank when hit | 4.988 | 5.287 |

## 8. 显存控制与常见坑

- RQ-VAE 使用 CPU，避免与 Qwen/vLLM 同进程。
- SFT 使用 NF4、BF16 compute、LoRA、gradient checkpointing 和小 batch。
- GRPO 使用双卡 FSDP actor + vLLM TP=2，max_num_seqs=4、response=96、动态 token batch、optimizer/reference offload、eager mode、关闭 prefix cache/chunked prefill。
- 保持 group_size=4；OOM 时先减小 prompt/response/cache 利用率，不改变实验定义。
- VERL checkpoint 补丁在 summon_full_params 上下文内部完成 LoRA CPU copy，并只保存 adapter。
- 训练与评估后检查 nvidia-smi、global_step_N/export/lora_adapter/adapter_model.safetensors 和 test metrics 是否同时存在。

## 9. 局限与后续实验

All_Beauty 只有 28 个 retained items，151 条 test 也不足以支持大规模论文结论。项目已经从单一 16 候选重排扩展为“全目录 Semantic-ID 召回 + Qwen/GRPO 重排”两阶段原型，但召回实验的 catalog 仍非常小。正式论文应使用 Video_Games、Movies_and_TV 或 Books，按用户 bootstrap 置信区间，并在 validation 上选择 checkpoint。后续还需研究分片 trie/ANN fallback、多正样本 session、候选约束 Qwen decoding、KL 系数和 reward 权重消融。

## 10. 可复现实验命令

~~~bash
bash scripts/prepare_data.sh configs/data_rqvae.yaml
bash scripts/train_sft.sh configs/sft_qlora_rqvae_1p5b.yaml
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_grpo.sh configs/grpo_verl_rqvae_1p5b_dual.yaml
bash scripts/export_grpo_adapter.sh \
  outputs/grpo/qwen2.5-1.5b-all-beauty-rqvae-dual global_step_40 \
  outputs/sft/qwen2.5-1.5b-all-beauty-rqvae
bash scripts/evaluate.sh configs/eval_rqvae_1p5b.yaml \
  --adapter-path outputs/grpo/qwen2.5-1.5b-all-beauty-rqvae-dual/global_step_40/export/lora_adapter
~~~
