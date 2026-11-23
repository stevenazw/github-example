# ts-factor-miner

对单个时序资产批量挖掘预测因子的系统（Python 包名 `tsminer`）。

## 1. 这是什么

目标资产可以是美国 10 年期国债期货 TY、EURUSD 这类单个日频时序。流程如下：

1. 从 DataBank 数据库取几百条原始序列；每条序列的可得性（值何时可知）由配置给定，没有规则的序列拒收。
2. 每条序列按固定的变换集合生成候选信号。
3. 每个候选在自己的有效数据内切成三段：样本内 IS 70%、观察期 OBS 15%、样本外 OOS 15%。
4. 第 1 步：在 IS 上做单因子检验，过门槛（PASS / WEAK_PASS）的候选留下。
5. 第 2 步：在观察期上筛选，方向或强度没有保持的候选被淘汰。
6. 第 3 步：对未被淘汰的候选按相关性去重，代表组成 shortlist，并给出 IC 衰减、信号自相关、逐年稳定性与连续仓位的回测读数。
7. OOS 只在 `confirm` 里打开一次。

## 2. 设计原则

- **三段切分，各有用途。** IS 用于单因子检验与过门槛；观察期可以用来淘汰和挑选；样本外只看一次，不得据此再做任何调整。
- **可得性是进入系统前给定的输入。** 系统不猜测、不套默认值：配置的 `[lags]` 里没有规则的序列被拒收（`no_availability_rule`）并在报告里列出。
- **逐候选的单因子检验。** 每个候选都是一次独立检验，没有跨候选的校正或组合。门槛（|z| ≥ 2.5 为 PASS，2.0–2.5 为 WEAK_PASS，< 2 为 REJECT）事先固定。
- **短历史候选在自身有效数据内检验并各自切分。** 候选序列的历史可能比标的短（例如标的 TY 从 1999 年开始，某序列 2019 年才有）。三段按这个候选自己的有效行切分，而不是按标的日期一刀切。
- **变换集合小而固定。** 日频 6 个、周频 4 个，不按变量类型区分，不做参数网格，不做公式搜索。
- **仓位用连续比例，并考虑 IC 衰减。** 报告 IC 衰减与信号自相关；慢信号不必每天调仓，用缓冲带控制换手。
- **防前视是数据层的职责。** 所有序列先 as-of 对齐到标的日历，再按可得性滞后；信号只用因果变换。
- **逐候选隔离。** 候选 j 在 `mine` 中的任何输出都不依赖它自己 OOS 行（行号 ≥ `e2[j]`）的数据取值；IS 阶段的统计量（t、null_sd、z、flags 中的 half_flip / xlag_weak）不依赖行号 ≥ `e1[j]` 的数据取值。`confirm` 是唯一打开 OOS 的入口，用 lock 文件保证只看一次。
- **判据在看数据前固定。** 门槛写在配置的 `[verdict]` 里。

## 3. 流程

```
databank.py        transforms.py      stats.py            pipeline.py                         confirm.py
数据库 --> Panel --> 候选信号 S --> HAC t / 零分布 sd --> IS 检验 -> OBS 筛选 -> 去重 runs/ --> OOS 确认
 (as-of 对齐+可得性)  (日频6/周频4)    (按各自的行集合)       (shortlist 等)                       (一次, lock)
```

### 3.1 时间约定

信号 `s_t` 在 t 日收盘已知；目标 `y_fwd[t] = r[t+1] / σ_t`，其中 σ_t 是只用 ≤ t 数据的 EWMA 波动率（`vol_halflife`，默认 20）。目标是 futures_chain 时，`r` 取数据库连续合约的 `ret`；是 spot 时取 `PX_LAST` 的对数差。

### 3.2 数据层与可得性输入（`databank.py`）

每条序列先 as-of 对齐到标的日历（取日期 ≤ 当天的最近一个观测，日频超过 5 个日历行、周频超过 10 个日历行视为过期，置 NaN），再向后移 `lag` 行。`lag` 的含义：t 日收盘的信号只能用日期 ≤ t−lag 的观测。数据库里的 `release_date` 不被使用。

可得性完全来自配置的 `[lags]`，是由研究者确认的输入，没有频率默认值：

- `chain`（必填）：期货链 `chain:*` 与 `carry:*` 的滞后。
- `[lags.class]`：键为 `"class_one/class_two"`。
- `[lags.ticker]`：按 ticker 精确指定，优先级高于 class。
- 优先级：ticker 精确 > class；都没有则 `resolve_lag` 返回 None，该序列拒收（`excluded` 里原因 `no_availability_rule`）。`[lags]` 里出现 `D` / `W` 这类旧键按未知键报错。
- 利差序列的两条腿各自对齐、滞后后再相减，滞后取两腿较大者；任一条腿没有可得性规则则 `build_panel` 直接报错（配置错误显性失败）。
- `Panel.meta` 的 `lag_rule` 列记录每条序列实际命中的规则键（`ticker:<ticker>`、`class:<class_one/class_two>`、`chain`；利差写两条腿的规则并用 `+` 连接）。

`configs/ty.toml` 的取值（`configs/eurusd.toml` 相同）。这些数值是研究者于 2026-11-23 确认的输入，不是系统推断：行情数据与期货连续合约不滞后，隐含波动率滞后 1 天；ETF 资金流应滞后 1 天（当前候选池未包含资金流字段，加入时需为其单独设规则）；其余类别沿用起草值。

| 序列类型 | lag | 依据 |
|---|---|---|
| `Market_Data/*`（债券、货币市场、互换/OIS、商品、股票、通胀挂钩、信用、MBS、外汇） | 0 | 行情数据，研究者确认不滞后 |
| `Market_Data/VOL`（衍生品隐含波动率） | 1 | 研究者确认滞后 1 天 |
| `Macro/Global`、`Macro/US` | 1 | 宏观日频指数，保持 1（研究者未改） |
| CFTC 持仓 (`Positioning/CFTC_COT`) | 4 | 周二数据，周五盘后发布，下周一可用 |
| Fed 资产负债表 (`FED/Balance_Sheet`) | 2 | 周三数据，周四发布，周五可用 |
| 一级交易商 (`FED/Primary_Dealer`) | 8 | 周三数据，下周四发布，保守取 8 |
| `NFCIINDX Index`（ticker 规则） | 6 | 周频发布 |
| 期货连续合约 `chain`、`carry` | 0 | 研究者确认不滞后 |

另外：不是日频也不是周频的序列被剔除（原因 `freq_not_daily_or_weekly`）；对齐后有限值少于 250 个的序列被剔除（`too_few_obs`）。`kind` 由数据库的 unit/data_type 推断：price 类取对数差，其余（level）取差分。近月连续合约（`MM_Future`、`Commodity_Future`）的换月跳空无法在现有数据上修正，已移出候选池。

### 3.3 候选信号（`transforms.py`）

6 个变换：`chg_1`、`chg_5`、`chg_20`、`chg_60`（k 日变化）和 `z_60`、`z_252`（相对滚动均值与标准差的 z-score）。k 日变化除以其自身在 `std_window`（默认 252）上的滚动标准差，最后截尾到 ±`clip`（默认 4）。日频序列用全部 6 个，周频序列只用 `chg_5`、`chg_20`、`chg_60`、`z_252`，因为周频序列再取 1 日变化或 60 日 z 没有意义。变换集合只由频率决定，不按变量类型区分。候选 id 形如 `<series>|<transform>`。所有变换只用 ≤ t 的数据。

### 3.4 逐候选三段切分（`pipeline.split_rows`）

设 `y_fwd` 为目标，`S` 为候选信号（全时间轴）。先只保留 `y_fwd` 有限的行（`rows`，行号 0..T−1 指这些行内的位置）。候选 j 的有效行是 `S[:, j]` 有限的行，共 `n_j` 个。

- `k1 = floor(n_j × is_frac)`，`k2 = floor(n_j × (is_frac + obs_frac))`；`p1`、`p2` 为第 k1+1、k2+1 个有效行的行号（不存在取 T）；
- 向下取整到 21 的倍数：`e1 = (p1 // 21) × 21`，`e2 = max(e1, (p2 // 21) × 21)`，这样切分点只取少数几个不同的值，零分布按 e1 分桶计算；
- IS = 有效行且行号 < e1；OBS = 有效行且 e1 ≤ 行号 < e2；OOS = 有效行且行号 ≥ e2。

每一行 p 的目标是 `r[p+1]`，行不相交即收益不相交。`split_rows` 对列用 cumsum 向量化。配置要求 `is_frac > 0`、`obs_frac > 0`、`is_frac + obs_frac < 1`（默认 0.70 / 0.15）。

### 3.5 第 1 步：IS 单因子检验（`stats.py`）

统计量是 y_fwd 对 s 的一元回归斜率的 Newey–West HAC t 值（`hac_lags`，默认 5），只在候选自己的 IS 行上算；IS 有效行数不足 `min_obs`（默认 250）的候选记为样本不足，不进入结果表（`n_skipped`）。

零分布标准差 `null_sd`：对候选 j，把 y 在 `[0, e1[j])` 上做循环平移（平移量在 `min_shift`..`e-min_shift` 间均匀取），重复 `n_boot` 次重算 t*，取其标准差。这保留各自的自相关，只打断对齐；厚尾信号的 t 天然更分散，除以自身的 null_sd 才公平。具有相同 `e1` 的候选共用一次计算。`e1 − 2×min_shift < 1` 的候选样本太短无法平移，同样计入 `n_skipped`。

`z = t / null_sd`；`p` 是由 |z| 按正态近似得到的双侧 p，仅供展示。判决：`|z| ≥ z_pass`（2.5）为 `PASS`，`|z| ≥ z_weak`（2.0）为 `WEAK_PASS`，否则 `REJECT`。门槛不随样本长度变，同样的 z 在短样本上对应更高的 IC（IC ≈ z ÷ √n_is）。

IS 上的 flags（不改变判决）：`half_flip`（IS 有效行按个数对半分，前后半段 t 异号）、`xlag_weak`（信号再多滞后一天后 |t| 低于 `extra_lag_t_min`、变号或非有限，说明信息集中在最近一天）。

### 3.6 第 2 步：观察期筛选（`pipeline.mine`）

只对第 1 步非 REJECT 的候选。在观察期上算 `t_obs, ic_obs`（至少 `OBS_MIN_OBS = 60` 个观测），`sign` 为 IS 上 t 的符号：

- `ic_obs_signed = sign × ic_obs`，`retention = ic_obs_signed / |ic|`（|IS 的 IC|）；
- `obs_status`：`t_obs` 非有限 → `INSUFFICIENT`（观察期样本不足，无法据此淘汰，保留并加 flag `obs_short`）；`retention ≥ obs_min_retention`（默认 0.5）→ `KEEP`；否则 `DROP`。

未被 DROP 的候选再算开发样本（DEV = IS ∪ OBS）上的带符号 `t_dev`，用于排序。

### 3.7 第 3 步：去重与 shortlist 读数

- **去重**：对未被 DROP 的候选按 `t_dev` 降序（NaN 排最后）贪心聚类。两两相关系数只用双方 DEV 行的交集，共同行少于 100 视为不相关；相关系数绝对值 ≥ `dedupe_corr` 的归入同一代表。各簇代表组成 shortlist，每个代表附 `cluster_size`。
- **IC 衰减**：h ∈ (1, 2, 5, 10, 20, 60)，`ic_h = sign × corr(s_t, y[t+h−1])`，只用 s_t 有限且 t 与 t+h−1 都在 DEV 内的行对。h=1 就是 DEV 上的带符号 IC。
- **信号自相关**：`ac_h = corr(s_t, s_{t−h})`（DEV 行），`signal_half_life = ln(0.5)/ln(ac_1)`（0 < ac_1 < 1；ac_1 ≤ 0 记 0；ac_1 ≥ 1 或非有限记 NaN）。
- **逐年 IC**：DEV 行按自然年分组，某年有效行 ≥ 60 才计入；`year_hit` 是带符号 IC > 0 的年份占比，`n_years` 是计入的年数。`n_years ≥ 3` 且 `year_hit < year_hit_min`（默认 0.6）加 flag `unstable`。
- **回测**：见 3.8。净 Sharpe ≤ 0 加 flag `cost_negative`。

### 3.8 仓位与回测

仓位是连续比例：`target_t = sign × s_t`（NaN 记 0）。缓冲带 `buffer`（`[position]`，默认 0.5）：`pos_0 = target_0`；若 `|target_t − pos_{t−1}| > buffer` 则 `pos_t = target_t`，否则 `pos_t = pos_{t−1}`。`buffer = 0` 时 pos 等于 target。回测评估区间从信号第一个有限值起：`gross = pos × y_fwd`；名义仓位 `pos / σ`，成本 `cost_per_side × |Δ名义仓位|`（首行用 |名义仓位₀|）；`net = gross − cost`。输出毛/净 Sharpe（均值÷标准差×√252）与 `turnover`（mean |Δpos|）。`mine` 在各自 DEV 行上算，`confirm` 在各自 OOS 行上算（从 OOS 首行重新起仓）。

### 3.9 读 IC 衰减与信号自相关

`ic_h1 … ic_h60` 看信息能持续多久：次日有效、之后迅速衰减到 0 的信号，应该快速调仓；长 h 上 IC 仍不低的，要同时看 `ac_h`。慢信号（如 `z_252`，`ac_1` 接近 1、`signal_half_life` 很长）的 IC 在长 h 上不低，部分只是因为信号本身变化慢，今天的信号与 h 天前几乎是同一个数，不等于信息持续。慢信号适合大的缓冲带、低换手；快信号换手高，要看净 Sharpe 与成本。

### 3.10 样本外确认（`confirm.py`）

shortlist 为空时直接返回 `NOTHING_TO_CONFIRM`，不建 lock。否则对每个 shortlist 候选取它自己的 OOS 行（至少 60 个观测）：`t_signed = sign × t_oos`，`ic_oos_signed = sign × ic_oos`，`retention_oos = ic_oos_signed / |IS 的 IC|`，`p_one = norm_sf(t_signed)`。三档结论：

- `CONFIRMED`：`p_one ≤ 0.10`；
- `CONSISTENT`：不满足上一条，但 `ic_oos_signed > 0` 且 `retention_oos ≥ obs_min_retention`（方向一致、强度保持，但统计上不够显著，OOS 短时常见）；
- `FAILED`：其余；t 非有限为 `INSUFFICIENT`。

没有跨候选的校正。完成后写入 `confirm.lock`。样本外只能看一次，之后不应再据其结果调整候选或参数。

## 4. 快速开始

**依赖**：Python ≥ 3.11（用到标准库 `tomllib`），`pip install -r requirements.txt`。

**数据前提**：先按 DataBank 的 START_HERE 说明把数据库还原到本地。配置里 `[databank]` 的 `toolkit_dir` 和 `cache_dir` 分别指向工具包（提供 `data_query`）与缓存目录（含 `current.json`），也可用环境变量 `DATABANK_TOOLKIT`、`DATABANK_CACHE` 覆盖。数据库本身和任何行情数据都不进仓库。

在项目根目录运行：

```bash
# 在 IS / OBS 上挖掘（耗时较长）；输出到 runs/<target>-<时间戳>/
python -m tsminer mine configs/ty.toml
python -m tsminer mine configs/eurusd.toml

# 可选参数
#   --run-dir DIR   指定输出目录
#   --panel DIR     复用已保存的 panel 目录（如 runs/<run_dir>/panel），不再查询数据库
python -m tsminer mine configs/ty.toml --run-dir runs/ty-test --panel runs/ty-old/panel

# 打开样本外确认 shortlist（只能一次）
python -m tsminer confirm runs/<run_dir>
# 已打开过则退出码为 2；确需重跑时必须给出理由，理由会记入 confirm_log.jsonl
python -m tsminer confirm runs/<run_dir> --force "理由"

# 测试
python -m pytest -q
```

**注意：`confirm` 会消耗样本外。** 如果打算把 shortlist 交给别的流程做最终检验（并由那个流程使用 OOS），就不要运行 `confirm`。

## 5. 输出文件

`runs/<name>/` 下（`runs/` 在 `.gitignore` 里，结果不进仓库）：

| 文件 | 内容 |
|---|---|
| `report.md` | 人类可读报告：结论、数据与样本、可得性输入、判定标准、shortlist、IC 衰减与信号自相关、提示与局限、下一步；confirm 之后追加样本外确认 |
| `run.json` | 机器可读摘要：配置快照、`dates`、`n_rows`、`n_series`、`n_candidates`、`n_tested`、`n_skipped`、`n_is_summary`、`funnel`（tested / is_pass / is_weak / obs_keep / obs_insufficient / obs_drop / shortlist）、shortlist；不含任何 OOS 取值 |
| `candidates.csv` | 第 1 步所有 t、null_sd 有效的候选，按 \|z\| 降序 |
| `shortlist.csv` | 去重后的代表候选，另含 `cluster_size`、回测、IC 衰减、信号自相关、逐年稳定性与描述列；可为空表 |
| `config.toml` | 本次运行所用配置的副本（`confirm` 从这里重建配置） |
| `panel/` | 保存的 Panel：`panel.pkl`（主数据，pickle）、`meta.csv`（含 `lag_rule`）、`excluded.csv`、`info.json` |
| `confirm.csv` / `confirm.json` | 样本外确认结果（`confirm` 之后才出现） |
| `confirm.lock` | OOS 已打开的标记 |
| `confirm_log.jsonl` | 每次 `--force` 的时间与理由 |

`panel.pkl` 含行情数据，且用 pickle 序列化，不要加载来路不明的文件。

`candidates.csv` 列：

| 列 | 含义 |
|---|---|
| `id`, `series`, `transform` | 候选 id（`series\|transform`）、原始序列、变换名 |
| `group`, `kind`, `freq` | 分组（`class_two`，或 Chain / Carry / Spread）、price/level、D/W |
| `n_is`, `is_start`, `is_end` | 该候选 IS 有效行数及首末日期 |
| `n_obs`, `n_oos` | 该候选 OBS、OOS 有效行的个数（只是个数，不含任何取值） |
| `t`, `null_sd`, `z` | IS 的 HAC t；该候选零分布 t 的标准差；z = t / null_sd |
| `ic` | IS 样本 IC（y 与 s 的相关） |
| `sign` | t 的符号，后续步骤与 `confirm` 用它定方向 |
| `p` | 由 \|z\| 按正态近似得到的双侧 p，仅供展示 |
| `verdict` | PASS / WEAK_PASS / REJECT |
| `ic_obs_signed`, `retention`, `obs_status` | 观察期带符号 IC、retention、KEEP / DROP / INSUFFICIENT；第 1 步 REJECT 的候选留空 |
| `t_dev` | DEV（IS+OBS）上的带符号 t，未被 DROP 的候选才有 |
| `t_h1`, `t_h2`, `t_xlag` | IS 前半段 t、后半段 t、多滞后一天的 t |
| `flags` | `half_flip` / `xlag_weak` / `obs_short`，shortlist 代表另有 `unstable` / `cost_negative`，分号分隔 |
| `rep` | 未被淘汰候选所属簇的代表 id |

`shortlist.csv` 在这些列之外还有 `cluster_size`、`sharpe_gross`、`sharpe_net`、`turnover`、`ic_h1`…`ic_h60`、`ac_1`…`ac_60`、`signal_half_life`、`year_hit`、`n_years` 与 `description`。

`confirm.csv` 列：`id, sign, n, oos_start, oos_end, t_signed, ic_oos_signed, retention_oos, p_one, verdict, sharpe_gross, sharpe_net`；`confirm.json` 另有 `n_tested, n_confirmed, n_consistent, n_failed, n_insufficient`。

## 6. 配置说明

参考 `configs/ty.toml`（futures_chain）和 `configs/eurusd.toml`（spot）。配置被严格校验，未知的节或键会报错。

| 节 / 键 | 含义 |
|---|---|
| `[target]` `name`, `kind`, `cost_per_side` | 必填。`kind = "futures_chain"` 时还需 `root`；`kind = "spot"` 时需 `ticker` 和 `field`（日收益为该序列的对数差）。`cost_per_side` 是单边成本（收益率单位） |
| `[databank]` `toolkit_dir`, `cache_dir` | 工具包与缓存目录，可被环境变量覆盖 |
| `[sample]` `start`, `is_frac`, `obs_frac` | `start` 为空则用标的全部历史；`is_frac`（默认 0.70）、`obs_frac`（默认 0.15）是每个候选各自有效行中 IS、OBS 的占比，其余为 OOS；要求 `is_frac > 0`、`obs_frac > 0`、`is_frac + obs_frac < 1` |
| `[universe]` `fields` | 字段优先级，每个 ticker 取它拥有的第一个 |
| `[universe]` `chains`, `carry_chains` | 期货链：生成 `chain:<root>`（return index，price）与 `carry:<root>`（日历价差 / 活跃价格，level）序列 |
| `[universe]` `exclude_tickers` | 从候选池剔除的 ticker |
| `[[universe.select]]` | 候选池选择，每条恰为三种形式之一：`class_one` / `class_two` / `class_three`（仅 `class_three` 可为列表）；`family` + `attrs`（如 CFTC）；`tickers`（显式列表） |
| `[[universe.spread]]` | `id`、`a`、`b`：生成 `spread:<id>` = a − b（level） |
| `[lags]` `chain`, `class`, `ticker` | 可得性输入，见 3.2。`chain` 必填；`class` 的键形如 `"class_one/class_two"`；没有频率默认值，出现 `D` / `W` 报错 |
| `[position]` `buffer` | 缓冲带宽度（默认 0.5，≥ 0），见 3.8 |
| `[signals]` `std_window`, `clip`, `vol_halflife` | 滚动标准化窗口、截尾幅度、EWMA 波动率半衰期 |
| `[stats]` `hac_lags`, `min_obs`, `n_boot`, `min_shift`, `seed` | HAC 滞后阶数、候选 IS 段最少观测数（默认 250）、零分布重复次数、循环平移的最小位移、随机种子 |
| `[verdict]` `z_pass`, `z_weak` | PASS / WEAK_PASS 门槛（默认 2.5 / 2.0），要求 `z_pass > z_weak > 0` |
| `[verdict]` `dedupe_corr`, `extra_lag_t_min` | 去重的相关阈值（默认 0.7）；`xlag_weak` 的 \|t\| 阈值（默认 2.0） |
| `[verdict]` `obs_min_retention`, `year_hit_min` | 观察期保留与 `confirm` 的 CONSISTENT 用的 retention 下限（默认 0.5）；`unstable` 的逐年命中率下限（默认 0.6） |

**候选池要按经济假设选。** 过门槛只代表单因子证据，候选越多，纯运气过门槛的候选也越多，最终靠观察期与样本外判定。选择必须在看数据之前做出。

## 7. 已知局限

- 可得性规则是输入假设：由研究者确认，系统不推断；没有处理政府停摆等造成的发布延迟（如 CFTC 报告推迟）。
- 月频、季频和不规则序列（多数宏观数据、拍卖、FOMC）未纳入：数据库里的 `release_date` 对历史不可靠，发布滞后不可得。
- 近月连续合约（`MM_Future`、`Commodity_Future`）因换月跳空无法在现有数据上修正，未纳入候选池。
- ACM 期限溢价被有意排除：它是全样本重估的模型序列，历史值含前视。
- spot 标的（EURUSD）的收益是现货对数收益，不含 carry。
- 回测是描述性的：成本是单边固定比例，缓冲带是固定参数，没有做 walk-forward。
- 门槛对所有候选一视同仁：短样本上同样的 z 需要更高的 IC，短样本候选的观察期与样本外也短，筛选与确认能力有限。
- 只检验"次日收益"这一个目标；更慢的信息通过 `chg_20`、`chg_60`、`z_252` 这类平滑变换进入，IC 衰减只作描述。
- 去重按其他候选的 DEV 状态排序，某候选的 `rep` / `cluster_size` 在极端情形下可能间接受到其他候选更晚数据的影响（与该候选自身的统计量无关）。
- EURUSD 在数据库里只从 2018 年开始，样本短、功效低。

## 8. 测试

测试编码的是设计意图，而不只是函数输出。`python -m pytest -q` 在项目根目录运行。最重要的几类：

- **因果性 / 无前视**：`test_causality`、`test_no_lookahead_invariance`、`test_forward_target_no_lookahead`、`test_lag_semantics_never_earlier`。
- **三段切分**：`test_split_rows*`（掩码不相交且并集等于有效行、e1/e2 是 21 的倍数、70/15/15 占比、只有后段有数据的列在自己的有效区间内切分）。
- **可得性输入**：`test_resolve_lag_priority_and_none`、`test_lags_section_is_an_input`、`test_build_panel_rejects_series_without_availability_rule`（需要本地数据库，否则 skip）。
- **逐候选隔离**：`test_isolation_nothing_from_oos`（行号 ≥ `e2` 之后的 target 与该候选原始序列换成随机数，candidates.csv 与 shortlist.csv 的数值列逐值不变）、`test_isolation_is_stats_ignore_obs_and_oos`（行号 ≥ `e1` 之后换掉，IS 统计量逐值不变），长历史与短历史候选各验证一次。
- **植入信号 / 观察期淘汰 / lock**：`test_planted_signal_survives_screens`、`test_planted_signal_confirmed`、`test_observation_period_drops_decayed_signal`、`test_lock_and_force`、`test_cli_locked_exit_code`、`test_nothing_to_confirm`。
- **仓位与回测**：`test_positions_*`、`test_backtest*`。
- **IC 衰减 / 信号自相关 / 逐年命中**：`test_ic_decay_*`、`test_autocorr_and_half_life_of_random_walk_zscore`、`test_year_hit_*`。
- **零分布标准差**：`test_null_sd_*`，以及 HAC t 的朴素对照与零假设校准。

集成测试 `test_build_panel_ty_integration` 等需要本地数据库，数据库不存在时会 skip。

## 9. 因子跟踪看板（`site`）

```bash
python -m tsminer site runs/ty-v5 runs/eurusd-v5 --out site-out
python -m http.server -d site-out        # 本地打开 index.html
```

每个 run 目录对应一个资产（两个 run 的 target 同名则报错退出，退出码 1）。该命令只读 run 目录，不运行 `mine` 或 `confirm`。

输出文件：

| 文件 | 内容 |
|---|---|
| `index.html` | 完整独立页面（单页，hash 路由：`#overview` 与每个资产 `#<slug>`） |
| `artifact.html` | 同一模板生成的片段版（无 doctype / html / head / body 标签），用于发布到托管环境 |
| `data/<slug>.json` | 每个资产一份（slug = target 小写），页面按需 `fetch`：shortlist 因子的净值、分段统计、公式、方向、样本外结论 |

**净值定义。** 对每个 shortlist 因子，取其有效行，在这些行上一次连续计算（缓冲带状态跨段延续）：`pos = positions(s, sign, buffer)`，`net = pos × y_fwd − cost_per_side × |Δ(pos/σ)|`（首行用 |pos₀/σ₀|）。标定 `k = (0.10/√252) / std(net，样本内行, ddof=1)`，`ret = k × net`，净值 = `1 + cumsum(ret)`（单利累计，不复利），即样本内年化波动标定为 10%。样本内 std 为 0 或非有限的因子不出图，记入 `skipped`。净值点的日期取收益实现日（信号行 p 的下一个日历行）；这与 `candidates.csv` 的 `is_end` 等日期（信号行的日期）晚一个日历行。每段统计有区间、天数、带符号 IC、净 Sharpe、累计收益（该段 ret 之和）、最大回撤。每个因子最多约 1500 个点，保留各段首末点。

**样本外何时显示。** 只有 run 目录里存在 `confirm.json`（即已运行过 `confirm`）才显示：此时输出三段净值并附 `confirm.csv` 的样本外结论。否则不读取、不计算、不输出任何样本外行，净值只到观察期末，页面标为"样本外未打开"。

