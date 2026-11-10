# MiniOneRec 多卡训练接线

这里是本仓库与 [MiniOneRec 官方代码](https://github.com/AkaliKong/MiniOneRec) 的接线说明。已核对官方 `convert_dataset.py`、`sft.py`、`rl.py`、`evaluate.py` 和 `rq/generate_indices.py` 的输入结构。本机没有 GPU，因此以下大模型训练命令尚未实际执行；运行前建议记录上游提交号和依赖版本。

## 0. 环境

```bash
git clone https://github.com/AkaliKong/MiniOneRec.git /path/to/MiniOneRec
cd /path/to/MiniOneRec
git rev-parse HEAD
conda create -n minionerec-mind python=3.11 -y
conda activate minionerec-mind
pip install -r requirements.txt
pip install sentence-transformers lightgbm
```

以下命令从**本仓库根目录**运行，使用绝对路径设置变量：

```bash
export UPSTREAM_DIR=/path/to/MiniOneRec
export DATA_DIR=/absolute/path/to/this-repo/data/processed
export BASE_MODEL=/path/to/Qwen2.5-1.5B
export SFT_OUT=/absolute/path/to/this-repo/runs/sft
export RL_OUT=/absolute/path/to/this-repo/runs/rl
export NUM_GPUS=4
```

## 1. 数据与新闻嵌入

先从 [MIND 官方页面](https://msnews.github.io/index.html)获取 MIND-small train/dev；将文件分别放在 `data/mind/train`、`data/mind/dev`，各目录应有 `news.tsv` 和 `behaviors.tsv`。

```bash
pip install -e .
python scripts/prepare_minionerec.py \
  --train-dir data/mind/train --dev-dir data/mind/dev --out "$DATA_DIR"
python scripts/embed_news.py \
  --item-json "$DATA_DIR/MIND.item.json" \
  --out "$DATA_DIR/MIND.emb.npy" \
  --model BAAI/bge-base-en-v1.5
```

`MIND.item.json` 的键是从 0 开始的连续整数；嵌入严格按该编号排序。与官方商品实验相比，新闻文本编码器换成英文新闻模型，这是**领域适配**，不是论文原始设置的精确复现。文本目录包含 train/dev 所有新闻，未使用 dev 点击标签构建 SID；MIND 没有新闻发布时间，因此这是转导式目录。

## 2. RQ-VAE 与 SID

```bash
cd "$UPSTREAM_DIR/rq"
python rqvae.py \
  --data_path "$DATA_DIR/MIND.emb.npy" \
  --ckpt_dir /absolute/path/to/this-repo/runs/rqvae \
  --lr 1e-3 --epochs 10000 --batch_size 20480
cd /absolute/path/to/this-repo
python scripts/generate_sid.py \
  --upstream-dir "$UPSTREAM_DIR" \
  --checkpoint /absolute/path/to/this-repo/runs/rqvae/best_collision_model.pth \
  --out "$DATA_DIR/MIND.index.json"
python scripts/prepare_minionerec.py \
  --train-dir data/mind/train --dev-dir data/mind/dev \
  --out "$DATA_DIR" --sid-index "$DATA_DIR/MIND.index.json"
```

检查 RQ-VAE 实际输出目录中的 checkpoint 路径后再填入 `--checkpoint`。`generate_sid.py` 读取官方 checkpoint 和 `RQVAE.get_indices`，对于重复编码追加确定性的末级 token，随后转换器会拒绝任何剩余冲突。这样约束解码的一个 SID 只对应一篇新闻。

准备好后检查：

```text
data/processed/MIND.item.json
data/processed/MIND.index.json
data/processed/news_to_item.json
data/processed/info/MIND.txt
data/processed/train/MIND.csv
data/processed/valid/MIND.csv
data/processed/test/MIND.csv
```

CSV 列与官方 `convert_dataset.py` 一致：`user_id`、`history_item_title`、`item_title`、`history_item_id`、`item_id`、`history_item_sid`、`item_sid`。`*.queries.jsonl` 保存导出行与 MIND 曝光 ID 的一一对应，不能排序或删除。

## 3. 官方 SFT / GRPO

官方脚本中的类别词典暂不含新闻；先在单独克隆的上游目录执行最小补丁：

```bash
python scripts/patch_upstream.py "$UPSTREAM_DIR" --check
python scripts/patch_upstream.py "$UPSTREAM_DIR"
bash scripts/run_sft.sh
```

补丁只给 `sft.py`、`rl.py`、`evaluate.py` 的 `category_dict` 增加 `MIND: news articles`，并移除 `evaluate.py` 硬编码 GPU 0 的语句；若上游代码结构变化，脚本会报错而不是盲目改动。SFT 用前段时间的 MIND train 训练，后段时间的 `valid` 用于模型验证和后续排序器训练。

若已有可用的 SFT checkpoint，再执行可选 GRPO：

```bash
bash scripts/run_rl.sh
```

请按实际显存调整 batch、长度、生成组数及 `NUM_GPUS`。官方 README 建议完整实验使用 4–8 张 A100/H100；本仓库的脚本参数是起点，不能代替显存核算。

## 4. 生成候选与导入

对排序训练集和最终测试集分别运行官方约束解码。若仅做 SFT 消融，将 `MODEL_DIR` 指向 SFT checkpoint；若评估 GRPO，将其指向 RL checkpoint。

```bash
export MODEL_DIR="$SFT_OUT"
SPLIT=valid bash scripts/run_eval.sh
SPLIT=test bash scripts/run_eval.sh
```

官方 `evaluate.py` 的 JSON 每行有 `predict` SID 列表，没有逐候选的可比较概率。导入器按原始 CSV 行序、`output` SID 和 `*.queries.jsonl` 对齐到曝光 ID；行数或目标不符会拒绝导入。训练排序器只用 `valid` 时段的**实际曝光候选标签**，`test` 为最终未参与训练的官方 dev 曝光。

```bash
newsrec-benchmark \
  --train-dir data/mind/train --dev-dir data/mind/dev \
  --upstream-data "$DATA_DIR" \
  --valid-generation-json "$DATA_DIR/valid/MIND.predictions.json" \
  --test-generation-json "$DATA_DIR/test/MIND.predictions.json" \
  --max-ranker-impressions 0 --max-eval-impressions 0 \
  --rank-backend lightgbm \
  -k 50 --report runs/report.json --model-out runs/ranker.txt
```

如需评估生成召回本身的真实推理耗时，应单独计时官方 `evaluate.py`。`newsrec-benchmark` 的 P95 仅包括本地热门/协同/内容检索与融合，不包含大模型生成时间。

## 已知边界

- 上游脚本可能继续演进，重新克隆后先运行补丁 `--check`，再用少量数据验证 SFT 和约束解码。
- 新闻点击在同一曝光内没有顺序；导出多正例时，它们共享曝光前历史。排序训练严格只使用有标签的曝光候选。
- 完整新闻目录与全量候选评估会占用较多 CPU/内存；先用默认 2000/500 曝光做流程检查，确认后传 `0` 执行全量。
- 如未拿到真实模型训练和评估结果，不应在简历中填写效果提升。
