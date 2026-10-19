# segmtree

可解释的无监督决策树分群库 —— 把一张"客户标签"表自动划分成若干人群，每个人群都由一条
简短、人类可读的规则链定义，全程不需要任何标签 y。

`segmtree` 实现的是**聚类决策树**（clustering tree）：不预测类别，而是自顶向下递归二分，
每一刀选择使"组内异质性下降最多"的单特征阈值规则，并按 Best-First（全局收益最大的叶子
先切）生长。可以理解为"没有 y 也能长的决策树"——同时给你划分结果和划分规则，这是
k-means / GMM 做不到的。

## 为什么选 segmtree？

| | k-means / GMM | segmtree |
|---|---|---|
| 人群定义 | 质心坐标 | `income > 50k AND is_active = 0` 式规则 |
| 新样本归类 | 最近质心 | 每条规则链仅几次阈值/成员判断 |
| 二值 + 连续混合特征 | 需要各种缩放技巧 | 归一化后同口径比较 |
| 内置自检 | 无 | 收益可加性 + 规则回放一致性校验 |
| 稳定性评估 | 自己想办法 | 一行调用的 Bootstrap ARI |

典型场景：带行为标签的客户分群、患者表型归类、问卷人群画像——一切"大量 0/1 标签 +
少量计数/金额列"的表。

## 安装

```bash
pip install segmtree
```

仅要求 Python ≥ 3.9 与 numpy。

## 快速上手

```python
import numpy as np
from segmtree import SegTree, extract_rules, profile_leaves, replay_check, bootstrap_stability

rng = np.random.default_rng(0)
X = np.vstack([
    np.hstack([rng.normal([0, -1, 2, .5], .5, (250, 4)), (rng.random((250, 3)) < .1)]),
    np.hstack([rng.normal([4, 3, -2, .5], .5, (200, 4)), (rng.random((200, 3)) < .5)]),
    np.hstack([rng.normal([-4, 2, 0, 8], .5, (150, 4)), (rng.random((150, 3)) < .9)]),
])

tree = SegTree(min_gain=0.02, min_leaf_frac=0.05, min_leaf_abs=10)
tree.fit(X)

print(f"{tree.n_leaves_} 类人群，异质性降低 {tree.reduction_:.0%}")

for rule in extract_rules(tree):
    print(rule)                      # #0: x1 <= -1.24 AND x0 <= 1.83 ...

profile_leaves(tree, X, top_k=3)     # LeafProfile：z 分数画像 + category_enrichment
replay_check(tree, X)                # True：规则回放与 predict 完全一致
bootstrap_stability(tree, X, n_replicates=12, random_state=0).mean   # 例如 0.98
```

取值只有 {0, 1} 的列会被自动识别为二值特征（只做一次"是/否"切分）；直接传 pandas
DataFrame 也可以，列名会自动带入规则文本。

## 类别特征

整数编码的类别列（如名义变量的 0..k-1 码）可通过 `categorical_features` 声明。
它们不按阈值切分，而是按**类别子集**划分，规则呈现为成员判断：

```python
tree = SegTree(categorical_features=["color"], max_categories=12).fit(df)
# 规则示例： #3: color in {1, 2} AND income > 42 ...
```

异质性贡献：类别列使用 **Gini 杂度相对根 Gini 的比值**，连续/二值列保持归一化
方差。Gini 只取决于类别*分布*，因此对编码做任何一一重映射都会得到等价的划分——
结果与编码方式无关。

注意：

- 编码必须为整数值；非数值标签请先因子化
  （`pd.factorize` 或 `df["color"].astype("category").cat.codes`）；
- 任意大小的编码都会被精确处理（内部 int64/float64），即使连续列选择
  `dtype=np.float32` 也不会合并；
- **候选子集**：默认 `categorical_splitter="auto"` 会根据基数与候选矩阵内存自动
  选择。列基数 `k <= max_categories` 时精确评估全部 `2^(k-1)-1` 种类别划分；超过
  时**自动降级**为 `"one_vs_rest"`（逐个评估"单类别 vs 其余"，代价随 k 线性增长），
  只发一条 `UserWarning`，不会让整个 `fit` 失败。显式写 `"exact"` 仍会抛出带有列名、
  候选数与预估内存的 `ValueError`。
- **按列指定策略**：`categorical_splitter={"city": "exact", "sku": "one_vs_rest"}`
  可以按特征名或索引混用两种策略，适合同一张表里既有 6 档年龄段又有 14 档渠道码的
  场景。
- 预测时未见过的编码由 `handle_unknown` 控制：`"complement"`（默认）确定性地
  落入补集一侧且 `replay_check` 依然严格一致；`"error"` 抛出带特征名与未知值
  的 `ValueError`。训练期类别集合通过 ``categories_`` 属性暴露；
- ``profile_leaves`` 通过 ``category_enrichment``（`(特征, 编码, 叶内占比,
  lift)`）报告类别结构，z 分数仅用于连续/二值特征。

## 方法原理

节点异质性是各特征杂度比值的加权平均：

```
H(S) = Σ_j w_j · Var_j(S)/Var_j(root) / Σ_j w_j        H(root) = 1
```

未传 `feature_weights` 时所有 `w_j = 1`。二值特征的伯努利方差 p(1−p) 天然适用
同一公式；声明为类别的列改用 Gini 杂度相对根 Gini 的比值。一刀切成 L/R 的收益：

```
Gain = H(S) − [ n_L/n_S · H(L) + n_R/n_S · H(R) ]
```

默认的 `hist` 路径先用全局分位数把每个连续特征切成至多 `max_bins` 个箱，节点内
按箱边界评估候选；候选先由关联度代理排序，再对前几个候选计算节点内**其他特征**的
真实异质性下降（详见下节）。显式 `splitter="exact"` 时才改回节点内分位数
（`quantiles`）与排序 + 前缀和的逐候选精确评估。frontier 放在以 *负 gain* 为键的
最小堆里，因此每次扩展都取当前全局收益最大的叶子（Best-First，同收益按插入顺序
稳定决胜）。满足任一条件即停止：最优刀收益低于 `min_gain`、子节点过小
（`max(min_leaf_abs, min_leaf_frac·n)`）、达到 `max_depth`，或叶数已达
`max_leaf_nodes`（每刀恰好新增一个叶子，预算耗尽后剩余候选节点直接成为叶子）。

两个开箱即用的可信度保证：

- **收益可加性** —— `total_gain_ == h_root_ − h_avg_`（浮点误差量级），任意合法
  权重下同样精确成立；
- **规则回放** —— `replay_check()` 从零重新求值每条叶子规则链，必须与 `predict()`
  逐行一致。

测试套件还包含与一个独立朴素参考实现逐位一致性的对照测试。

## 大数据与性能

**默认所有数据规模都走 `splitter="hist"`**。旧的精确算法只作为显式开启的
`splitter="exact"` 对照/验证模式保留，不再自动选用；它的每节点代价为
`O(F² · n_node)`，工程上不推荐用于生产。

| 实现 | 适用场景 | 每节点代价 | 内存 |
|---|---|---|---|
| `"hist"`（默认） | 所有数据规模 | `O(F · n_node)` | `n×F` 个值 + 每值 1 字节分箱码（>256 箱时 2 字节）+ 拟合期 `O(F²)` 关联矩阵 |
| `"exact"` | 仅显式开启：与原始算法对照、复现旧结果 | `O(F² · n_node)` | 分块缓冲，默认 ≤192 MiB |

`exact` 主要用于与独立朴素参考实现的逐位一致性测试，以及需要原始准则语义的
特殊场景；正常使用保持默认即可。

`hist` 路径先用**关联度代理**在全部候选里排序（数值列用 Pearson r²、类别-数值用
相关比 η²、类别-类别用 Cramér's V²；在最多 5 万行上估计），再对前 4 个候选计算
**节点内其他特征的真实异质性下降**，并扣除随机切分带来的 `1/n_eff` 噪声底。这样：

- 不会因为"任何连续列中位数切一刀都掉 64% 方差"而把纯噪声切成一堆叶子；
- 不会在同一人群内部继续过分裂；
- 多列且样本量足够时，纯噪声数据会稳定停在根节点（1 个叶子）。单列数据没有
  跨特征佐证，会像 exact 路径一样按自身方差切分（见下方边界说明）。

阈值 `min_gain` 由 `min_gain_mode` 控制：默认 `"auto"` 在 hist 路径用 `sqrt(F)`
归一，因此**新增弱特征不会再把树卡在根节点**；只有显式 `splitter="exact"` 时才
保持旧的（按特征数平均的）语义。需要更激进/更保守时可显式设 `"total"` /
`"normalized"`。

实测（单机；峰值 RSS 在拟合期间采样，**包含调用方的输入矩阵**；最大规模用
`dtype=np.float32`）：

| 数据 | 路径 | 耗时 | 峰值内存 | 叶子 |
|---|---|---|---|---|
| 20 万 × 30 | hist | 2.5 s | 263 MB | 4 |
| 6 万 × 300（3 个信号列） | hist | 5.5 s | 489 MB | 4 |
| 30 万 × 301（3 个信号列） | hist | 29 s | 1,015 MB | 4 |
| **82.9 万 × 301（3 个信号列，f32）** | hist | **84 s** | **2,439 MB（输入占 1,152 MB）** | 4 |
| 6 万 × 300（纯噪声） | hist | 3.3 s | 497 MB | 1 |

倒数第二行对应用户报告的原始规模：原实现在该数据上直接 OOM；现在约 1.4 分钟、
拟合自身额外内存约 1.3 GB，并且只切在 3 个信号列中的 2 个上。

小/中数据同样不需要 exact：1,200×6 上 hist 约 0.011 s / 5 个叶子，exact 约
0.089 s / 69 个叶子；20,000×50 上 hist 约 0.39 s / 7 个叶子，exact 约 8.2 s /
65 个叶子（exact 在宽数据上更容易把噪声切成大量叶子）。这正是默认全部走 hist 的
原因。

其他相关参数：

- `max_bins=32`：hist 路径每个连续特征的分位数箱数，箱越多阈值越细、每节点略慢；
- `random_state`：固定关联度子采样与并列收益的随机决胜；不传时默认完全可复现；
- `verbose=1`：每切一刀打印深度、节点大小、特征、收益、子节点大小与耗时，长任务
  不再"看起来卡死"；
- `bootstrap_stability(..., subsample=0.5)`：每次重采样只训练一半规模，控制稳定性
  评估的成本。

**选型提示**：保持默认 `splitter="hist"` 即可；超过百万行/数百列时建议显式设
`dtype=np.float32`（内部值与输入各占一份，内存约减半）。只有需要与旧结果逐位
对照时才使用 `splitter="exact"`，并请先估算 `4·F²·n` 的工作量。

**边界说明**：hist 路径要求切分能得到*其他特征*的佐证。若数据中只有一个特征携带
结构、其余特征与它完全独立，hist 会保守地停在根节点（这种结构在无监督下与噪声
不可区分）。此时可显式使用 `splitter="exact"`，并按业务需要调高 `min_gain`。

## 特征加权

业务上常常"某些列更重要"。`feature_weights` 接受长度等于特征数的一维数值序列、
按列名索引的 dict（未列出的列权重为 1），或默认 `None`（等权重）：

```python
tree = SegTree(feature_weights={"income": 4.0, "churn_flag": 2.0}).fit(df)
```

权重缩放各列对异质性与收益的贡献——它并不禁止在零权重列上切分：只要零权重列能很好
地分开其他被加权的特征，仍可作为代理变量被选中。

## 导出与持久化

```python
d = tree.export_dict()        # 可直接 JSON 序列化的嵌套 dict
import json; json.dumps(d)    # 开箱即用

print(tree.export_text())     # 缩进文本：分支条件、gain、样本数、叶子编号
print(tree.export_graphviz()) # Graphviz DOT 字符串（零依赖）

tree2 = SegTree.from_dict(d)  # 从 export_dict 重建可 predict 的树
```

split 节点包含 `feature/feature_index/op/threshold|categories/gain/n_samples/
left/right`；leaf 节点包含 `leaf_id/n_samples/fraction/heterogeneity/conditions`。
不暴露任何私有对象。`from_dict` 可完整还原预测能力，适合把规则部署到生产评分端。

## API 一览

```python
SegTree(min_gain=0.01, min_leaf_frac=0.01, min_leaf_abs=10, max_depth=None,
        quantiles=8, binary_features="auto", categorical_features=None,
        max_categories=12, handle_unknown="complement", feature_names=None,
        dtype=np.float64, feature_weights=None, max_leaf_nodes=None,
        categorical_splitter="auto", splitter="hist", max_bins=32,
        min_gain_mode="auto", random_state=None, verbose=0)
```

| 成员 | 用途 |
|---|---|
| `.fit(X)` / `.fit_predict(X)` / `.predict(X)` | sklearn 风格训练与归类 |
| `.score(X)` | 异质性降低比例（可直接用于 `GridSearchCV`） |
| `.transform(X)` | 叶编号 one-hot 编码（Pipeline 中间步骤） |
| `.apply(X)` | 叶编号（sklearn `apply` 约定） |
| `.labels_`, `.n_leaves_`, `.leaves_`, `.cuts_` | 拟合结构 |
| `.reduction_`, `.h_avg_`, `.total_gain_`, `.summary()` | 质量统计 |
| `.feature_importances_` | 按特征归一化的切分收益 |
| `.feature_weights_` | 生效的逐特征权重 |
| `.splitter_`, `.association_` | 实际使用的搜索实现与逐特征关联度 |
| `.root_variances_`, `.root_ginis_` | 根节点逐特征原始方差 / Gini（`h_root_` 是实际根异质性，通常为 1.0；`h_root_normalized_` 才是归一化基准） |
| `.export_dict()` / `.export_text()` / `.export_graphviz()` | 公开树导出 |
| `SegTree.from_dict(d)` | 从 `export_dict` 重建可 predict 的树 |
| `extract_rules(tree)` | 返回 `Rule` 列表（`str(rule)` 即规则链文本） |
| `replay_labels(tree, X)` / `replay_check(tree, X)` | 规则回放校验 |
| `profile_leaves(tree, X, top_k=3)` | 每类 `LeafProfile` 画像 |
| `bootstrap_stability(tree, X, ...)` | Bootstrap ARI（mean/std） |
| `adjusted_rand_score(a, b)` | 纯 numpy 实现的 ARI |

调参建议：

- 想要更少、更大的人群 → 提高 `min_gain`（如 0.02~0.05）、减少 `quantiles`；
  已知目标人群数时**直接设 `max_leaf_nodes`**；
- 大数据/宽表 → 保持默认 `splitter="hist"`；想要更细或更粗的阈值可调 `max_bins`；
  需要旧版阈值语义时设 `min_gain_mode="normalized"`；
- 规则想更短、结果想更稳 → 设 `max_depth`（4~6）；
- 小数据集（< 500 行）→ 可进一步降低 `min_leaf_abs`（默认 10）；
- 有业务侧重点 → 用 `feature_weights` 强调关键列，而不是删掉其余特征；
- Bootstrap ARI 偏低 → 通常减少分位数、提高 `min_gain`、限深或限叶数即可稳定结构。

## 与相关工具的比较

- **先 k-means 再用 sklearn 决策树拟合**（代理树）：两步走，规则可能与聚类结果矛盾；
  segmtree 把切分质量与规则一致性放进同一个目标。
- **forest-clustering / URF**：基于森林邻近度的聚类，精度好但人群本身不可直接解释。
- **CUBT（R 包）**：学术上最接近的方法；segmtree 在纯 numpy 包内补齐了 Best-First
  生长、混合类型归一化、可加性自检与 Bootstrap 稳定性。

## 开发

```bash
git clone <your-fork-url> && cd segmtree
pip install -e .[dev]
pytest                       # 运行测试
ruff check src tests         # lint
ruff format --check src tests  # 格式
```

## 许可证

MIT — 见 [LICENSE](LICENSE)。

[English](README.en.md)
