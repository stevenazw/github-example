# ts-factor-miner

对单个时序资产批量挖掘预测因子的系统（Python 包名 `tsminer`）。

## 1. 这是什么

目标资产可以是美国 10 年期国债期货 TY、EURUSD 这类单个日频时序。流程如下：

1. 从 DataBank 数据库取几百条原始序列。
2. 每条序列按固定的变换集合生成候选信号。
3. 在 train 段上逐个检验候选对次日收益的预测力。
4. 用多重检验控制"同批测了几千个"的问题。
5. 给出 shortlist 和组合因子。
6. 在 holdout 上只确认一次。

## 2. 设计原则

- **默认结论是"没有发现"。** 单资产日频数据的样本量有限。以 TY 为例，train 段约 5400 天、族大小约 2500 时，单个候选要通过 FWER 门槛大约需要 |z| ≥ 4.2，相当于日频 IC ≈ 0.057。现实因子的 IC 多在 0.02–0.04，单独看几乎不可能通过。这是样本量算术，不是方法失败。因此系统同时给出三层读数：单个候选的判决、分组/全族 omnibus 检验（这批候选整体是否含有信息）、经验贝叶斯后验与组合因子。
- **变换集合小而固定（6 个）。** 不做参数网格，不做公式搜索。每多一个候选都会抬高门槛。
- **防前视是数据层的职责。** 所有序列先 as-of 对齐到标的日历，再按可得性滞后；信号只用因果变换。
- **train/holdout 隔离。** `mine` 的任何输出不依赖 holdout；`confirm` 是唯一打开 holdout 的入口，用 lock 文件保证只看一次。
- **判据在看数据前固定。** 门槛写在配置的 `[verdict]` 里。

## 3. 流程

```
databank.py        transforms.py      stats.py            pipeline.py          confirm.py
数据库 --> Panel --> 候选信号 S --> HAC t / 零分布 / EB --> 判决+输出 runs/ --> holdout 确认
 (as-of 对齐+滞后)  (6 个因果变换)   (只用 train 段)        (shortlist 等)      (一次, lock)
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

### 3.4 检验统计量（`stats.py`）

统计量是 y_fwd 对 s 的一元回归斜率的 Newey–West HAC t 值（`hac_lags`，默认 5）。

零分布：把 y 相对信号做循环平移（平移量在 `min_shift`..`T-min_shift` 间均匀取），重复 `n_boot` 次。这样保留各自的自相关和信号间的相关性，只打断对齐。每个候选除以自身零分布的标准差得到 z，因为厚尾信号的 t 天然更分散，不标准化会不公平。

### 3.5 判决与提示（`pipeline.py`）

- `PASS`：single-step maxT 的族错误率 `p_fwer ≤ fwer_alpha`。
- `WEAK_PASS`：BH `q_bh ≤ fdr_q`。
- 否则 `REJECT`。

`flags` 列给出两个提示，不改变判决：

- `half_flip`：train 前后半段的 t 异号。
- `xlag_weak`：信号再多滞后一天后 |t| 低于 `extra_lag_t_min` 或变号，说明信息集中在最近一天，信号短命，或取值时刻晚于假设。

### 3.6 Omnibus、经验贝叶斯与组合因子

- **omnibus**：全族与各分组（按 `class_two`，加上 Chain / Carry / Spread）的 mean(z²) 对比零分布；分组之间做 Holm 校正（`groups.csv`）。
- **经验贝叶斯**：两组混合模型 z ~ π0·N(0, v0) + (1−π0)·N(0, v0 + n·τ²)。v0 是零分布下 z² 的均值，n = (z/ic)²。每个候选得到 `p_signal`（= 1 − local fdr）、收缩后的 `post_ic`、`p_pos`（P(IC>0)）。
- **去冗余**：对非 REJECT 的候选按 |z| 从高到低贪心聚类，信号相关系数绝对值 ≥ `dedupe_corr` 的归入同一代表；各簇代表组成 shortlist。
- **组合因子**：全族候选按 `post_ic` 加权求和（缺失值按 0），权重在 train 上定死。仅当 τ² > 0 且 π0 < 1 时有定义（`run.json` 的 `composite_defined`）。

### 3.7 holdout 确认（`confirm.py`）

shortlist 每个候选按 train 上的符号做单边 HAC t 检验，对 p 做 Holm 校正；组合因子单独做一个单边检验（不并入 Holm）。结论按 p 分档：≤ 0.05 为 `CONFIRMED`，≤ 0.10 为 `WEAK`，否则 `NOT_CONFIRMED`。完成后写入 `confirm.lock`。

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
| `report.md` | 人类可读报告：结论、门槛、shortlist、watchlist、分组 omnibus；confirm 之后追加 holdout 结果 |
| `run.json` | 机器可读摘要：日期切分、族大小、零分布门槛 `hurdle`、omnibus、EB 参数、判决计数、shortlist |
| `candidates.csv` | 族内全部候选，按 \|z\| 降序 |
| `shortlist.csv` | 去冗余后的代表候选，另含描述与 train 段回测列 |
| `watchlist.csv` | |z| 最大的 200 个候选去冗余后取前 20 个，不一定通过检验，仅供参考 |
| `groups.csv` | 分组 omnibus：`n_cand`、`mean_z2`、`null_mean_z2`、`p`、`p_holm` |
| `config.toml` | 本次运行所用配置的副本（`confirm` 从这里重建配置） |
| `panel/` | 保存的 Panel：`panel.pkl`（主数据，pickle）、`meta.csv`、`excluded.csv`、`info.json` |
| `confirm.csv` / `confirm.json` | holdout 确认结果（`confirm` 之后才出现） |
| `confirm.lock` | holdout 已打开的标记 |
| `confirm_log.jsonl` | 每次 `--force` 的时间与理由 |

`panel.pkl` 含行情数据，且用 pickle 序列化，不要加载来路不明的文件。

`candidates.csv` 主要列：

| 列 | 含义 |
|---|---|
| `id`, `series`, `transform` | 候选 id（`series\|transform`）、原始序列、变换名 |
| `group`, `kind`, `freq` | 分组（`class_two`，或 Chain / Carry / Spread）、price/level、D/W |
| `n` | 有效观测数 |
| `t`, `null_sd`, `z` | HAC t；该候选零分布 t 的标准差；z = t / null_sd |
| `ic` | 样本 IC（y 与 s 的相关） |
| `sign` | t 的符号，`confirm` 用它定方向 |
| `p_cal`, `q_bh` | 由 \|z\| 按正态近似得到的双侧 p；BH 校正后的 q |
| `p_fwer` | single-step maxT 的族错误率 p 值 |
| `verdict` | PASS / WEAK_PASS / REJECT |
| `p_signal`, `shrink`, `post_ic`, `p_pos` | EB：非零后验概率、收缩系数、后验 IC、P(IC>0) |
| `t_h1`, `t_h2`, `t_xlag` | 前半段 t、后半段 t、多滞后一天的 t |
| `flags` | `half_flip` / `xlag_weak`，分号分隔 |
| `rep` | 非 REJECT 候选所属簇的代表 id |

`shortlist.csv` 和 `watchlist.csv` 在这些列之外还有 `sharpe_gross`、`sharpe_net`、`turnover`、`ac1` 与 `description`。

## 6. 配置说明

参考 `configs/ty.toml`（futures_chain）和 `configs/eurusd.toml`（spot）。配置被严格校验，未知的节或键会报错。

| 节 / 键 | 含义 |
|---|---|
| `[target]` `name`, `kind`, `cost_per_side` | 必填。`kind = "futures_chain"` 时还需 `root`；`kind = "spot"` 时需 `ticker` 和 `field`（日收益为该序列的对数差）。`cost_per_side` 是单边成本（收益率单位） |
| `[databank]` `toolkit_dir`, `cache_dir` | 工具包与缓存目录，可被环境变量覆盖 |
| `[sample]` `start`, `holdout_frac` | `start` 为空则用标的全部历史；`holdout_frac` 是尾部 holdout 占比（TY 为 0.2，EURUSD 为 0.3） |
| `[universe]` `fields` | 字段优先级，每个 ticker 取它拥有的第一个 |
| `[universe]` `chains`, `carry_chains` | 期货链：生成 `chain:<root>`（return index，price）与 `carry:<root>`（日历价差 / 活跃价格，level）序列 |
| `[universe]` `exclude_tickers` | 从候选池剔除的 ticker |
| `[[universe.select]]` | 候选池选择，每条恰为三种形式之一：`class_one` / `class_two` / `class_three`（仅 `class_three` 可为列表）；`family` + `attrs`（如 CFTC）；`tickers`（显式列表） |
| `[[universe.spread]]` | `id`、`a`、`b`：生成 `spread:<id>` = a − b（level） |
| `[lags]` `D`, `W`, `class`, `ticker` | 见 3.2。`class` 的键形如 `"class_one/class_two"` |
| `[signals]` `std_window`, `clip`, `vol_halflife` | 滚动标准化窗口、截尾幅度、EWMA 波动率半衰期 |
| `[stats]` `hac_lags`, `min_obs`, `n_boot`, `min_shift`, `seed` | HAC 滞后阶数、候选最少观测数、零分布重复次数、循环平移的最小位移、随机种子 |
| `[verdict]` `fwer_alpha`, `fdr_q`, `dedupe_corr`, `extra_lag_t_min` | PASS 的 FWER 水平、WEAK_PASS 的 FDR 水平、去冗余相关阈值、`xlag_weak` 的 \|t\| 阈值 |

**缩小族以降低门槛。** 按经济假设只选相关类别，而不是把能取到的都放进来。族从约 2500 降到 50 时，FWER 门槛大约从 |z| 4.2 降到 3.2，对应所需 IC 明显下降。这个选择必须在看数据之前做出。

## 7. 已知局限

- 月频、季频和不规则序列（多数宏观数据、拍卖、FOMC）未纳入：数据库里的 `release_date` 对历史不可靠，发布滞后不可得。
- 可得性滞后是按类别的常规发布日程设定的，没有处理政府停摆等造成的发布延迟（如 CFTC 报告推迟）。
- ACM 期限溢价被有意排除：它是全样本重估的模型序列，历史值含前视。
- spot 标的（EURUSD）的收益是现货对数收益，不含 carry。
- 回测指标（Sharpe 毛/净、换手）只是描述性的，成本模型是单边固定比例。
- BH 的 p 值用正态近似；候选之间高度相关，FDR 控制是近似的。FWER 的 maxT 不受此影响。
- 只检验"次日收益"这一个 horizon；更慢的信息通过 `chg_20`、`chg_60`、`z_252` 这类平滑变换进入。
- EURUSD 在数据库里只从 2018 年开始，样本短、功效低。

## 8. 测试

测试编码的是设计意图，而不只是函数输出。`python -m pytest -q` 在项目根目录运行。最重要的几类：

- **因果性 / 无前视**：`test_causality`、`test_no_lookahead_invariance`、`test_forward_target_no_lookahead`、`test_lag_semantics_never_earlier`（改动未来数据不应改变过去的信号或对齐结果）。
- **holdout 隔离**：`test_holdout_isolation`、`test_prepare_split`。
- **FWER 控制**：`test_fwer_control_pure_noise`、`test_null_calibration_*`、`test_pure_noise_no_pass`（纯噪声模拟下不应出现 PASS）。
- **植入信号可检出**：`test_planted_signal_detected`、`test_planted_signal_confirmed`。
- **lock**：`test_lock_and_force`、`test_cli_locked_exit_code`。

集成测试 `test_build_panel_ty_integration` 需要本地数据库，数据库不存在时会 skip。
