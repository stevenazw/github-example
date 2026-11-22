# ts-factor-miner

对单个时序资产批量挖掘预测因子的系统（Python 包名 `tsminer`）。

## 1. 这是什么

目标资产可以是美国 10 年期国债期货 TY、EURUSD 这类单个日频时序。流程如下：

1. 从 DataBank 数据库取几百条原始序列。
2. 每条序列按固定的变换集合生成候选信号。
3. 每个候选作为独立的单因子检验，在它自己与标的重叠的 train 样本上检验对次日收益的预测力。
4. 过门槛（PASS / WEAK_PASS）的候选留下。
5. 对留下的候选按相关性去重，代表组成 shortlist。
6. 在 holdout 上只确认一次。

## 2. 设计原则

- **逐候选的单因子检验。** 每个候选都是一次独立检验，没有"同批"概念，也没有跨候选的校正或组合。门槛（|z| ≥ 2.5 为 PASS，2.0–2.5 为 WEAK_PASS，< 2 为 REJECT）事先固定，不因测了多少个候选而打折。通过门槛只是"留下"，成立与否由 holdout 判定。
- **短历史候选在自身重叠样本上检验，并各自切分。** 候选序列的历史可能比标的短（例如标的 TY 从 1999 年开始，某序列 2019 年才有）。每个候选只用它自己与标的重叠的有效行，train / holdout 按这个候选自己的样本切分，而不是按标的日期一刀切。
- **变换集合小而固定（6 个）。** 不做参数网格，不做公式搜索。
- **防前视是数据层的职责。** 所有序列先 as-of 对齐到标的日历，再按可得性滞后；信号只用因果变换。
- **train/holdout 隔离（逐候选）。** 候选 j 在 `mine` 中的任何统计量都不依赖它自己 holdout 行（行号 ≥ `end_row[j]`）的数据，包括 y 在这些行的取值；`confirm` 是唯一打开 holdout 的入口，用 lock 文件保证只看一次。
- **判据在看数据前固定。** 门槛写在配置的 `[verdict]` 里。

## 3. 流程

```
databank.py        transforms.py      stats.py              pipeline.py            confirm.py
数据库 --> Panel --> 候选信号 S --> HAC t / 零分布 sd --> 逐候选检验+去重 runs/ --> holdout 确认
 (as-of 对齐+滞后)  (6 个因果变换)   (只用各自 train 行)    (shortlist 等)           (一次, lock)
```

### 3.1 时间约定

信号 `s_t` 在 t 日收盘已知；目标 `y_fwd[t] = r[t+1] / σ_t`，其中 σ_t 是只用 ≤ t 数据的 EWMA 波动率（`vol_halflife`，默认 20）。目标是 futures_chain 时，`r` 取数据库连续合约的 `ret`；是 spot 时取 `PX_LAST` 的对数差。

### 3.2 数据层与可得性滞后（`databank.py`）

每条序列先 as-of 对齐到标的日历（取日期 ≤ 当天的最近一个观测，日频超过 5 个日历行、周频超过 10 个日历行视为过期，置 NaN），再向后移 `lag` 行。数据库里的 `release_date` 不被使用，可得性完全由配置的 lag 决定。滞后的优先级是：`[lags.ticker]` > `[lags.class]` > 频率默认。数值取自 `configs/ty.toml`：

| 序列类型 | lag | 依据 |
|---|---|---|
| 日频默认 (`D`) | 1 | 保守：不假设 x 的取值时刻早于标的收盘 |
| 周频默认 (`W`) | 6 | 保守覆盖"下周才发布"的情形 |
| CFTC 持仓 (`Positioning/CFTC_COT`) | 4 | 周二数据，周五盘后发布，下周一可用 |
| Fed 资产负债表 (`FED/Balance_Sheet`) | 2 | 周三数据，周四发布，周五可用 |
| 一级交易商 (`FED/Primary_Dealer`) | 8 | 周三数据，下周四发布，保守取 8 |

这些都是保守假设。若确知某序列在标的收盘前已可得，可在 `[lags.ticker]` 里把它设为 0。期货链 `chain:*`、`carry:*` 用 `D`；利差序列的两条腿各自对齐、滞后后再相减，滞后取两腿较大者。

另外：不是日频也不是周频的序列被剔除（原因 `freq_not_daily_or_weekly`）；对齐后有限值少于 250 个的序列被剔除（`too_few_obs`）。`kind` 由数据库的 unit/data_type 推断：price 类取对数差，其余（level）取差分。

### 3.3 候选信号（`transforms.py`）

6 个变换：`chg_1`、`chg_5`、`chg_20`、`chg_60`（k 日变化）和 `z_60`、`z_252`（相对滚动均值与标准差的 z-score）。k 日变化除以其自身在 `std_window`（默认 252）上的滚动标准差，最后截尾到 ±`clip`（默认 4）。日频序列用全部 6 个，周频序列只用 `chg_5`、`chg_20`、`chg_60`、`z_252`，因为周频序列再取 1 日变化或 60 日 z 没有意义。候选 id 形如 `<series>|<transform>`。所有变换只用 ≤ t 的数据。

### 3.4 逐候选样本与切分（`pipeline.split_rows`）

设 `y_fwd` 为目标，`S` 为候选信号（全时间轴）。先只保留 `y_fwd` 有限的行（`rows`，行号 0..T−1 指这些行内的位置）。候选 j 的有效行是 `S[:, j]` 有限的行，共 `n_j` 个。切分点：

- `k_j = floor(n_j × (1 − holdout_frac))`，`p_j` 为第 k_j+1 个有效行的行号（k_j = n_j 时取 T）；
- 向下取整到 21 的倍数：`end_row[j] = (p_j // 21) × 21`，这样切分点只取少数几个不同的值，零分布按切分点分桶计算；
- train 行 = 有效行且行号 < `end_row[j]`；holdout 行 = 有效行且行号 ≥ `end_row[j]`。

每一行 p 的目标是 `r[p+1]`，行不相交即收益不相交。`split_rows` 对列用 cumsum 向量化。

### 3.5 检验统计量（`stats.py`）

统计量是 y_fwd 对 s 的一元回归斜率的 Newey–West HAC t 值（`hac_lags`，默认 5），只在候选自己的 train 行上算；train 有效行数不足 `min_obs`（默认 250）的候选记为样本不足，不进入结果表（`n_skipped`）。

零分布标准差 `null_sd`：对候选 j，把 y 在 `[0, end_row[j])` 上做循环平移（平移量在 `min_shift`..`e-min_shift` 间均匀取），重复 `n_boot` 次重算 t*，取其标准差。这保留各自的自相关，只打断对齐；厚尾信号的 t 天然更分散，除以自身的 null_sd 才公平。具有相同 `end_row` 的候选共用一次计算。`end_row − 2×min_shift < 1` 的候选样本太短无法平移，同样计入 `n_skipped`。

`z = t / null_sd`；`p` 是由 |z| 按正态近似得到的双侧 p，仅供展示。

### 3.6 判决、提示与去重（`pipeline.py`）

- **判决**：`|z| ≥ z_pass`（2.5）为 `PASS`，`|z| ≥ z_weak`（2.0）为 `WEAK_PASS`，否则 `REJECT`；z 非有限为 `REJECT`。门槛不随样本长度变，同样的 z 在短样本上对应更高的 IC（IC ≈ z ÷ √n_train）。
- **flags**（逐候选，不改变判决）：
  - `half_flip`：把候选自己的 train 有效行按个数对半分，前后半段的 t 都有限且异号。
  - `xlag_weak`：信号再多滞后一天后（仍用该候选自己的 train 行）|t| 低于 `extra_lag_t_min`、变号或非有限，说明信息集中在最近一天，信号短命，或取值时刻晚于假设。
- **去重**：对所有非 REJECT 候选按 |z| 从高到低贪心聚类。两两相关系数只用双方 train 行的交集，共同行少于 100 视为不相关；相关系数绝对值 ≥ `dedupe_corr` 的归入同一代表。各簇代表组成 shortlist，每个代表附 `cluster_size`（它代表的非 REJECT 候选数，含自己）。
- **描述性回测**：shortlist 候选在各自 train 行上算 Sharpe（毛/净）、换手、一阶自相关。

### 3.7 holdout 确认（`confirm.py`）

shortlist 为空时直接返回 `NOTHING_TO_CONFIRM`，不建 lock。否则对每个 shortlist 候选取它自己的 holdout 行，按 train 上的符号做单边 HAC t 检验（至少 60 个观测）：`t_signed = sign_train × t_hold`，`p_one = norm_sf(t_signed)`。`p_one ≤ 0.05` 为 `CONFIRMED`，`≤ 0.10` 为 `WEAK`，否则 `NOT_CONFIRMED`；t 非有限为 `INSUFFICIENT`。结果含 holdout 的 `n`、起止日期、IC 与 Sharpe（毛/净）。没有跨候选的校正。完成后写入 `confirm.lock`。短样本候选的 holdout 也短，确认能力有限。

## 4. 快速开始

**依赖**：Python ≥ 3.11（用到标准库 `tomllib`），`pip install -r requirements.txt`。

**数据前提**：先按 DataBank 的 START_HERE 说明把数据库还原到本地。配置里 `[databank]` 的 `toolkit_dir` 和 `cache_dir` 分别指向工具包（提供 `data_query`）与缓存目录（含 `current.json`），也可用环境变量 `DATABANK_TOOLKIT`、`DATABANK_CACHE` 覆盖。数据库本身和任何行情数据都不进仓库。

在项目根目录运行：

```bash
# 在 train 段挖掘（耗时较长）；输出到 runs/<target>-<时间戳>/
python -m tsminer mine configs/ty.toml
python -m tsminer mine configs/eurusd.toml

# 可选参数
#   --run-dir DIR   指定输出目录
#   --panel DIR     复用已保存的 panel 目录（如 runs/<run_dir>/panel），不再查询数据库
python -m tsminer mine configs/ty.toml --run-dir runs/ty-test --panel runs/ty-old/panel

# 打开 holdout 确认 shortlist（只能一次）
python -m tsminer confirm runs/<run_dir>
# 已打开过则退出码为 2；确需重跑时必须给出理由，理由会记入 confirm_log.jsonl
python -m tsminer confirm runs/<run_dir> --force "理由"

# 测试
python -m pytest -q
```

**注意：`confirm` 会消耗 holdout。** 如果打算把 shortlist 交给别的流程做最终检验（并由那个流程使用 holdout），就不要运行 `confirm`。

## 5. 输出文件

`runs/<name>/` 下（`runs/` 在 `.gitignore` 里，结果不进仓库）：

| 文件 | 内容 |
|---|---|
| `report.md` | 人类可读报告：结论、数据与样本、判定标准、shortlist、提示与局限、下一步；confirm 之后追加 holdout 确认 |
| `run.json` | 机器可读摘要：配置快照、`dates`（样本起止）、`n_rows`、`n_series`、`n_candidates`、`n_tested`、`n_skipped`、`n_train_summary`、判决计数、shortlist；不含任何 holdout 取值 |
| `candidates.csv` | 全部已检验候选，按 \|z\| 降序 |
| `shortlist.csv` | 去重后的代表候选，另含 `cluster_size`、描述与 train 回测列；可为空表 |
| `config.toml` | 本次运行所用配置的副本（`confirm` 从这里重建配置） |
| `panel/` | 保存的 Panel：`panel.pkl`（主数据，pickle）、`meta.csv`、`excluded.csv`、`info.json` |
| `confirm.csv` / `confirm.json` | holdout 确认结果（`confirm` 之后才出现） |
| `confirm.lock` | holdout 已打开的标记 |
| `confirm_log.jsonl` | 每次 `--force` 的时间与理由 |

`panel.pkl` 含行情数据，且用 pickle 序列化，不要加载来路不明的文件。

`candidates.csv` 列：

| 列 | 含义 |
|---|---|
| `id`, `series`, `transform` | 候选 id（`series\|transform`）、原始序列、变换名 |
| `group`, `kind`, `freq` | 分组（`class_two`，或 Chain / Carry / Spread）、price/level、D/W |
| `n_train`, `train_start`, `train_end` | 该候选 train 有效行数及首末日期 |
| `n_holdout` | 该候选 holdout 有效行的个数（只是个数，不含任何取值） |
| `t`, `null_sd`, `z` | HAC t；该候选零分布 t 的标准差；z = t / null_sd |
| `ic` | train 样本 IC（y 与 s 的相关） |
| `sign` | t 的符号，`confirm` 用它定方向 |
| `p` | 由 \|z\| 按正态近似得到的双侧 p，仅供展示 |
| `verdict` | PASS / WEAK_PASS / REJECT |
| `t_h1`, `t_h2`, `t_xlag` | train 前半段 t、后半段 t、多滞后一天的 t |
| `flags` | `half_flip` / `xlag_weak`，分号分隔 |
| `rep` | 非 REJECT 候选所属簇的代表 id |

`shortlist.csv` 在这些列之外还有 `cluster_size`、`sharpe_gross`、`sharpe_net`、`turnover`、`ac1` 与 `description`。

## 6. 配置说明

参考 `configs/ty.toml`（futures_chain）和 `configs/eurusd.toml`（spot）。配置被严格校验，未知的节或键会报错。

| 节 / 键 | 含义 |
|---|---|
| `[target]` `name`, `kind`, `cost_per_side` | 必填。`kind = "futures_chain"` 时还需 `root`；`kind = "spot"` 时需 `ticker` 和 `field`（日收益为该序列的对数差）。`cost_per_side` 是单边成本（收益率单位） |
| `[databank]` `toolkit_dir`, `cache_dir` | 工具包与缓存目录，可被环境变量覆盖 |
| `[sample]` `start`, `holdout_frac` | `start` 为空则用标的全部历史；`holdout_frac` 是每个候选各自有效行中尾部 holdout 的占比（TY 为 0.2，EURUSD 为 0.3） |
| `[universe]` `fields` | 字段优先级，每个 ticker 取它拥有的第一个 |
| `[universe]` `chains`, `carry_chains` | 期货链：生成 `chain:<root>`（return index，price）与 `carry:<root>`（日历价差 / 活跃价格，level）序列 |
| `[universe]` `exclude_tickers` | 从候选池剔除的 ticker |
| `[[universe.select]]` | 候选池选择，每条恰为三种形式之一：`class_one` / `class_two` / `class_three`（仅 `class_three` 可为列表）；`family` + `attrs`（如 CFTC）；`tickers`（显式列表） |
| `[[universe.spread]]` | `id`、`a`、`b`：生成 `spread:<id>` = a − b（level） |
| `[lags]` `D`, `W`, `class`, `ticker` | 见 3.2。`class` 的键形如 `"class_one/class_two"` |
| `[signals]` `std_window`, `clip`, `vol_halflife` | 滚动标准化窗口、截尾幅度、EWMA 波动率半衰期 |
| `[stats]` `hac_lags`, `min_obs`, `n_boot`, `min_shift`, `seed` | HAC 滞后阶数、候选 train 段最少观测数（默认 250）、零分布重复次数、循环平移的最小位移、随机种子 |
| `[verdict]` `z_pass`, `z_weak` | PASS / WEAK_PASS 门槛（默认 2.5 / 2.0），要求 `z_pass > z_weak > 0` |
| `[verdict]` `dedupe_corr`, `extra_lag_t_min` | 去重的相关阈值（默认 0.7）；`xlag_weak` 的 \|t\| 阈值（默认 2.0） |

**候选池要按经济假设选。** 过门槛只代表单因子证据，候选越多，纯运气过门槛的候选也越多，最终靠 holdout 判定。选择必须在看数据之前做出。

## 7. 已知局限

- 月频、季频和不规则序列（多数宏观数据、拍卖、FOMC）未纳入：数据库里的 `release_date` 对历史不可靠，发布滞后不可得。
- 可得性滞后是按类别的常规发布日程设定的，没有处理政府停摆等造成的发布延迟（如 CFTC 报告推迟）。
- ACM 期限溢价被有意排除：它是全样本重估的模型序列，历史值含前视。
- spot 标的（EURUSD）的收益是现货对数收益，不含 carry。
- 回测指标（Sharpe 毛/净、换手）只是描述性的，成本模型是单边固定比例。
- 门槛对所有候选一视同仁：短样本上同样的 z 需要更高的 IC，短样本候选的 holdout 也短，确认能力有限。
- 只检验"次日收益"这一个 horizon；更慢的信息通过 `chg_20`、`chg_60`、`z_252` 这类平滑变换进入。
- EURUSD 在数据库里只从 2018 年开始，样本短、功效低。

## 8. 测试

测试编码的是设计意图，而不只是函数输出。`python -m pytest -q` 在项目根目录运行。最重要的几类：

- **因果性 / 无前视**：`test_causality`、`test_no_lookahead_invariance`、`test_forward_target_no_lookahead`、`test_lag_semantics_never_earlier`（改动未来数据不应改变过去的信号或对齐结果）。
- **逐候选切分**：`test_split_rows*`（掩码不相交、切分点是 21 的倍数、holdout 占比、只有后段有数据的列在自己的有效区间内切分）。
- **逐候选隔离**：`test_per_candidate_isolation`（把行号 ≥ `end_row` 之后的 target 与该候选原始序列换成随机数，该候选的所有统计量逐值不变，长历史与短历史候选各验证一次）。
- **短历史候选可检验**：`test_short_history_candidate_is_tested`、`test_short_history_candidate_confirmed`。
- **零分布标准差**：`test_null_sd_*`，以及 HAC t 的朴素对照与零假设校准。
- **植入信号 / lock / 空 shortlist**：`test_planted_signal_confirmed`、`test_lock_and_force`、`test_cli_locked_exit_code`、`test_nothing_to_confirm`。

集成测试 `test_build_panel_ty_integration` 需要本地数据库，数据库不存在时会 skip。
