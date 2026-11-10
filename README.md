# MiniOneRec × MIND：生成式召回与曝光感知排序

面向搜广推算法实习的可复现实验项目：将 [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 的语义 ID 生成召回适配到 [MIND-small](https://msnews.github.io/index.html) 新闻推荐，再与新闻内容召回融合，并在真实曝光候选上训练点击排序器。

**当前状态：**数据解析、时间切分、内容召回、动态融合、排序、指标和官方格式转换均已实现并通过合成数据端到端测试。真实 MIND 数据和 MiniOneRec 多卡训练须在用户服务器上运行；本仓库不提供未经实测的效果数字，也不把合成数据测试当作模型效果。

## 项目结构

```text
MIND train.behaviors.tsv ──时间切分──► MiniOneRec 训练集 ──► 语义 ID 生成候选
                                    └► 后期排序训练集 ────────┐
新闻 title + abstract ──TF-IDF 全库召回───────────────────► 双路候选融合
真实曝光候选与历史特征 ───────────────────────────────────► Pairwise 点击排序
MIND dev.behaviors.tsv ──────────────────────────────────► 最终离线测试
```

| 模块 | 实现 | 可验证输出 |
|---|---|---|
| 数据 | MIND 新闻与曝光解析、严格时间切分、点击训练样本 | 无未来曝光标签、无未曝光负例 |
| 生成召回 | 新闻文本嵌入 → 官方 RQ-VAE → SID → 官方 SFT/可选 GRPO → 约束解码 | Top-K SID 映射回新闻 ID |
| 传统基线 | 热门新闻、历史到点击的 ItemCF | 全库 Top-K |
| 内容召回 | 标题和摘要的稀疏 TF-IDF，最近历史加权 | 全库 Top-K |
| 融合 | 固定配额与“历史长度 + 两路前列重合度”动态配额，RRF 排序 | Recall@10/50 消融 |
| 排序 | 曝光候选的内容相似度、类别偏好、标题词重合、生成候选名次；标准库 Pairwise，或可选 LightGBM LambdaRank | 曝光组 AUC/MRR/nDCG@5/10 |

官方 `evaluate.py` 目前只保存候选名次，没有可靠的逐候选生成概率。因此本项目以**两路召回重合度**作为动态配额的置信代理；导入器给候选的分数是负名次，并明确不将其解释为模型概率。

## 1 分钟运行：无需 GPU

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
newsrec-benchmark \
  --train-dir examples/tiny_mind/train \
  --dev-dir examples/tiny_mind/dev \
  -k 3
```

`examples/tiny_mind` 是手写的**合成数据**，只检验程序链路和指标定义。结果里的 AUC、Recall 或毫秒数不得写进简历。

## 用 MIND-small 跑真实实验

1. 从 [MIND 官方页面](https://msnews.github.io/index.html) 下载 `MINDsmall_train.zip`、`MINDsmall_dev.zip`，分别解压到 `data/mind/train` 和 `data/mind/dev`。不使用无点击标签的官方 test 做监督评估。
2. 在多卡环境安装可选依赖 `python -m pip install -e '.[embedding,rank]'`，再建立新闻目录与交互文件：

```bash
python scripts/prepare_minionerec.py \
  --train-dir data/mind/train --dev-dir data/mind/dev --out data/processed
```

3. 按 [多卡流程](docs/UPSTREAM.md) 编码新闻、训练官方 RQ-VAE、生成唯一 SID、训练 MiniOneRec 并导出 `valid` / `test` 候选。
4. 生成离线报告：

```bash
newsrec-benchmark \
  --train-dir data/mind/train --dev-dir data/mind/dev \
  --upstream-data data/processed \
  --valid-generation-json data/processed/valid/MIND.predictions.json \
  --test-generation-json data/processed/test/MIND.predictions.json \
  --max-ranker-impressions 0 --max-eval-impressions 0 \
  --rank-backend lightgbm \
  -k 50 --report runs/report.json --model-out runs/ranker.txt
```

若尚未训练 MiniOneRec，省略生成结果参数即可运行热门、ItemCF、内容召回与点击排序基线；报告仅展示实际可用的方法。

## 评估口径

- **召回**：在全新闻库检索，按每条 held-out 曝光中真实点击的新闻计算 Recall@K。用户已读新闻排除。未曝光新闻无标签，不能视为负样本。
- **排序**：只对 `behaviors.tsv` 记录的曝光候选计算组内 AUC、MRR、nDCG@5/10。AUC 跳过全正或全负曝光组并报告有效组数。
- **切分**：MIND train 按时间分成前 80% 生成模型训练、后 20% 排序训练；MIND dev 留作最终测试。训练集与排序集的同时间戳不会拆开。
- **长尾**：按生成模型训练阶段的点击计数划分，另报长尾已点击新闻的召回；具体阈值保存在报告中。
- **限制**：MIND 新闻表没有精确发布时间，目录以 train + dev 的新闻文本构建，属于仅使用文本的转导式目录。离线 Python 延迟不等于线上服务延迟。

建议的实验矩阵和简历写法见 [实验记录](docs/EXPERIMENTS.md)。

## 相关工作

- [MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation](https://arxiv.org/abs/2510.24431)：SID、SFT、GRPO 主干。
- [MIND: A Large-scale Dataset for News Recommendation](https://aclanthology.org/2020.acl-main.331/)：新闻与曝光日志。
- [CAPE](https://arxiv.org/abs/2610.29298)：生成召回和协同补召回融合的相关工作。本项目定位为公开曝光数据上的系统适配与实证分析，不声称首创多路融合。
- [UNIQUE](https://arxiv.org/abs/2609.23718)：生成式召排一体化的相关研究；本项目采用分阶段排序以便清晰评估曝光标签。

本仓库的代码由项目作者实现；[MiniOneRec 官方实现](https://github.com/AkaliKong/MiniOneRec) 需单独克隆并遵守其 Apache-2.0 许可。MIND 数据按其官方使用条件获取，仓库不重新分发原始数据或模型权重。
