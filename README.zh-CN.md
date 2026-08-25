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
| 新样本归类 | 最近质心 | 至多几条阈值判断即可回放 |
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

profile_leaves(tree, X, top_k=3)     # 每类与全体差异最大的特征（z 分数）
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
- 节点内会精确评估全部 `2^(k-1)-1` 种类别划分，请保持
  `k <= max_categories`（稀有水平建议先归并）；高基数列更适合 one-hot；
- 预测时未见过的编码由 `handle_unknown` 控制：`"complement"`（默认）确定性地
  落入补集一侧且 `replay_check` 依然严格一致；`"error"` 抛出带特征名与未知值
  的 `ValueError`。训练期类别集合通过 ``categories_`` 属性暴露；
- ``profile_leaves`` 通过 ``category_enrichment``（`(特征, 编码, 叶内占比,
  lift)`）报告类别结构，z 分数仅用于连续/二值特征。

## 方法原理

节点异质性 = 各特征组内方差除以根方差后的平均：

```
H(S) = (1/m) · Σ_j Var_j(S) / Var_j(root)          H(root) = 1
```

二值特征的伯努利方差 p(1−p) 天然适用同一公式，因此两类特征可在同一口径下比较。
一刀切成 L/R 两块的收益：

```
Gain = H(S) − [ n_L/n_S · H(L) + n_R/n_S · H(R) ]
```

候选阈值取各特征在节点内的分位数（`quantiles=8` 即 5%~95% 共 8 个；二值特征仅 0.5
一刀）。每个特征用排序 + 前缀和 O(n log n) 批量评估全部候选，堆维护全局最优叶子，
Best-First 生长。满足任一条件即停止：最优刀收益低于 `min_gain`、子节点过小
（`max(min_leaf_abs, min_leaf_frac·n)`）、达到 `max_depth`。

两个开箱即用的可信度保证：

- **收益可加性** —— `total_gain_ == h_root_ − h_avg_`（浮点误差量级），精确核算这棵树
  解释掉了多少总异质性；
- **规则回放** —— `replay_check()` 从零重新求值每条叶子规则链，必须与 `predict()`
  逐行一致。

测试套件还包含与一个独立朴素参考实现逐位一致性的对照测试。

## API 一览

```python
SegTree(min_gain=0.01, min_leaf_frac=0.01, min_leaf_abs=50, max_depth=None,
        quantiles=8, binary_features="auto", categorical_features=None,
        max_categories=12, handle_unknown="complement", feature_names=None,
        dtype=np.float64)
```

| 成员 | 用途 |
|---|---|
| `.fit(X)` / `.fit_predict(X)` / `.predict(X)` | sklearn 风格训练与归类 |
| `.labels_`, `.n_leaves_`, `.leaves_`, `.cuts_` | 拟合结构 |
| `.reduction_`, `.h_avg_`, `.total_gain_`, `.summary()` | 质量统计 |
| `extract_rules(tree)` | 返回 `Rule` 列表（`str(rule)` 即规则链文本） |
| `replay_labels(tree, X)` / `replay_check(tree, X)` | 规则回放校验 |
| `profile_leaves(tree, X, top_k=3)` | 每类 z 分数画像 |
| `bootstrap_stability(tree, X, ...)` | Bootstrap ARI（mean/std） |
| `adjusted_rand_score(a, b)` | 纯 numpy 实现的 ARI |

调参建议：

- 想要更少、更大的人群 → 提高 `min_gain`（如 0.02~0.05）或减少 `quantiles`；
- 规则想更短、结果想更稳 → 设 `max_depth`（4~6）；
- 小数据集（< 5000 行）→ 降低 `min_leaf_abs`（默认 50 是为大数据设的下限）；
- Bootstrap ARI 偏低 → 通常减少分位数、提高 `min_gain` 或限深即可稳定结构。

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
pytest            # 运行测试
ruff check src tests
```

## 许可证

MIT — 见 [LICENSE](LICENSE)。

[English](README.md)
