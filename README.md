# e-means-validation

Meta FAIR が発表したクラスタリングアルゴリズム **e-means** について、従来の k-means に対する高速化・品質改善の主張が実際に再現するのかを、**CPU 上で独立に**検証するリポジトリです。

> 目的は e-means の優秀さを証明することではなく、主張が **どの条件で再現し、どの条件で再現しないか** を実測に基づいて明らかにすることです。

## 1. 背景と目的

公式実装・論文は次のように主張しています（公式 README より）。

- Lloyd の k-means と比べ、**エポック数・実時間とも約10倍の高速化**、かつ **より低い損失**
- 追加のハイパーパラメータ不要（バッチサイズ・ステップ幅はスケーリング則で自動決定）

本リポジトリで検証する問い:

1. e-means は従来の k-means（Faiss Lloyd 法）より本当に高速か
2. 同等のクラスタリング品質に、より少ないエポック数で到達できるか
3. データの分布・次元・クラスタ数によって結果が変わるか
4. 実務で導入を検討する価値があるか

参考資料:

- 公式実装: <https://github.com/facebookresearch/faiss/tree/main/contrib/e_means>
- 論文: <https://arxiv.org/abs/2611.09501>
- 公式再現コード: <https://github.com/facebookresearch/faiss/tree/main/benchs/e_means>

## 2. e-means の概要

e-means はミニバッチ k-means の一種で、各バッチで集めた割当て数 `n_j` と座標和 `s_j` を **指数移動平均（EMA）** で重心に反映します。

- バッチサイズ `B` と初期ステップ幅 `α0` は、データ数 `n`・クラスタ数 `k`・エポック数 `T` からのスケーリング則で自動決定（`batch_law="auto"` は d≤16 で quality、それ以外で throughput）
- ステップ幅はコサインスケジュールで減衰
- 使われなくなったクラスタは、大きなクラスタの分割で「復活」させる（d≤2 では既定で無効）
- 最終エポックでは割当ての非加重平均で重心を確定
- バッチ全体＝全データ・ステップ1のとき、1回の Lloyd 反復と一致

CPU では `faiss.contrib.e_means.NumpyEmeans`（最近傍探索に `faiss.knn` を使用）、GPU では PyTorch + Triton/CuTe カーネルの `Emeans` を使います。**本リポジトリは CPU の `NumpyEmeans` のみを対象とします。**

## 3. 検証環境の構築

前提: Windows 11 + WSL2（Ubuntu）、[uv](https://docs.astral.sh/uv/)、git。GPU・Docker は不要です。

```bash
uv sync                          # Python 3.11 と依存（faiss-cpu, numpy, pytest）を uv.lock どおりに復元
bash scripts/setup_faiss.sh      # 公式 Faiss ソースを FAISS_COMMIT の SHA で faiss-src/ に取得（追加で必要な手順）
uv run pytest                    # 基本テスト
```

- Python のバージョンは `.python-version`（3.11）、依存バージョンは `uv.lock` に固定されています。
- 検証時の主な実バージョン: Python 3.11.17 / faiss-cpu 1.16.0 / numpy 2.4.6 / pytest 9.2.0。
- `scripts/setup_faiss.sh` は固定 SHA の 1 コミットだけを浅く取得します（約 19MB）。`faiss-src/` は Git 管理外です。
- 公式コードは site-packages にコピーせず、`benchmark.py` が `faiss-src/contrib/e_means` を直接読み込みます。読み込み元・SHA・ソースのハッシュは毎回検証され、`metadata.json` に記録されます。固定 SHA と異なる、または公式コードに変更がある場合は実行を停止します。
- 補足: faiss-cpu 1.16.0 の wheel にも `faiss.contrib.e_means` が同梱されており、固定 SHA の `*.py` と同一であることを確認済みです（`bundled_wheel_e_means_identical`）。それでも本検証では、管理された固定ソースから読み込みます。

### 実行する外部コード・依存

| 種類 | 内容 | ライセンス |
|---|---|---|
| PyPI | faiss-cpu, numpy（実行時）、pytest（開発時） | MIT / BSD-3-Clause / MIT |
| Git | facebookresearch/faiss @ `FAISS_COMMIT`（`contrib/e_means` のみ import） | MIT |

ネットワーク通信は `uv sync` と `scripts/setup_faiss.sh` のみで、ベンチマーク実行中に外部送信はありません（公式 e-means コードにも通信処理がないことを確認済み）。

## 4. ベンチマーク実行手順

```bash
# 動作確認（小規模、数秒）
uv run python benchmark.py --profile quick --threads 4

# 性能評価（200k × 1024 クラスタ、4 スレッドで十数分）
uv run python benchmark.py --profile medium --threads 4

# 同一初期重心で比較（初期化の違いによる交絡を除去）
uv run python benchmark.py --profile quick --threads 4 --init shared

# 同一 seed を複数回実行して計時ノイズを確認
uv run python benchmark.py --profile quick --threads 4 --repeats 3

# 特定データセット・シード
uv run python benchmark.py --profile quick --datasets imbalanced --seeds 10 11 12 13 14

# 集計とレポートの再生成
uv run python make_report.py results/<run-name>
uv run python make_scaling_table.py reports/<run1> reports/<run2> ...  # 規模を振った比較表
uv run python make_tolerance_table.py reports/<run1> reports/<run2> ...  # 許容幅を変えた再読（補足）

# 公開実データ（SIFT1M・DEEP1M の 1M 行）の取得と変換（約 550MB、data/ に保存、Git 管理外）
uv run python prepare_real_data.py
uv run python benchmark.py --profile medium --input data/sift1m.npy --threads 4 --quality-only

# 品質（MSE）だけを見る実験（他の作業と並行してよい。レポートに時間・倍率を出さない）
uv run python benchmark.py --profile medium --datasets imbalanced --quality-only \
    --emeans-unused-threshold 0 --run-name q_imb_norevival
```

### 速度計測の運用

**速度（時間・倍率）の計測は、他の作業がない時間帯（夜間など）に `scripts/run_speed_suite.sh` で行います。** IDE の操作や他プロセスとの CPU の取り合いで、時間が数十％〜数倍ずれることを実測で確認しているためです（[docs/methodology.md](docs/methodology.md) の「計時の診断」）。

```bash
bash scripts/run_speed_suite.sh                       # quick（3 反復）→ medium、4 スレッド
bash scripts/run_speed_suite.sh --profiles medium --max-wait-min 120
bash scripts/run_scaling_suite.sh                     # 規模を振る検証（k・n・d を 1 つずつ、約 3 時間）
bash scripts/run_scaling_suite.sh --points "k4096 n1m" # 一部の点だけ
bash scripts/run_speed_suite.sh --profiles "" --real "sift1m deep1m"  # 実データ（medium 規模）だけ
bash scripts/run_night_suite.sh                       # 未計測の夜間課題を順に（下記）
```

`run_night_suite.sh` は、未計測の速度の課題を安い順に 1 つずつ実行します（途中で止まっても、終わった分のレポートは揃います）。

1. 低次元で throughput 則を明示した規模の点 `d2tp`・`d16tp`（4 スレッド、約 10 分）。公式デフォルトは d ≤ 16 で quality 則を選ぶので、既存の `d2`・`d16` と比べて、e-means が低次元で遅い原因がバッチの小ささかを切り分けます。
2. Medium をスレッド数 6 → 4 → 2 → 1 で計測します。4 スレッドは計測済みですが、スレッド数の比較を同じ夜の計測どうしで行うために再計測します（品質は前回とビット単位で一致するはずなので、その確認にもなります）。6 は本環境の論理コア数のすべてなので、他のプロセスとの取り合いが起きやすくなります。

各段は元のスイートと同じ確認（未コミットの変更がないこと、ロードアベレージ、取り合いの数）をします。スレッド数をそろえた比較表は `make_scaling_table.py` に Medium の `speed_*_medium_t{1,2,4,6}` レポートを渡して作れます（行ラベルにスレッド数が出ます）。

- 未コミットの変更があると実行しません（結果をコミットに対応させるため）。
- 1 分ロードアベレージが 1.0 未満になるまで待ってから開始します（`--max-wait-min` 分待っても下がらなければ中止）。
- 各実行で CPU の取り合いが検出された回数を表示し、検出があればレポート冒頭に警告を出します。
- 自動実行する場合の例（Windows タスク スケジューラから WSL を起動）:
  `wsl.exe -d Ubuntu -- bash -lc "cd ~/Work_2404_001/e-means-validation && bash scripts/run_speed_suite.sh"`。
  実行中は Windows がスリープしない設定にしてください。

日中に行う実験は `--quality-only` を付け、品質（MSE・空クラスタ・到達エポック数）だけを評価します。

主なオプション: `--quality-only`、`--ntrain` / `--ntest` / `--dim` / `--k`（プロファイルの値を上書き。規模を振る検証用）、`--emeans-batch-law auto|throughput|quality`、`--emeans-unused-threshold`（e-means のクラスタ復活の閾値。既定は公式デフォルト）、`--imbalance-exponent`（既定 1.3 = キット）、`--kmeans-niters`（既定 5 10 25）、`--kmeans-inits` / `--emeans-inits`（既定 `random kmeanspp`）、`--emeans-epochs`（既定 1 2 3 5 10）、`--baseline-niter`（25）、`--tolerance`（0.01）、`--order shuffled|fixed`、`--no-warmup`、`--input data.npy`（公開データのみ）、`--allow-unpinned`。

計測の公平性のために行っていること:

- 同一の学習・評価データ、同一の k・次元数で比較
- `OMP/OPENBLAS/MKL_NUM_THREADS` と `faiss.omp_set_num_threads` でスレッド数を固定
- 計時は `train()` 呼び出しのみ（データ生成・評価は含まない）
- 学習データと評価データ（未学習データ）を分離
- 各データセットで計測外のウォームアップを実施
- 各 (dataset, seed) 内で構成の実行順をシャッフル
- 複数シード（quick 3 / medium 5）で中央値とばらつきを評価
- Faiss k-means の 256 点/クラスタのサンプリング上限を解除（全件学習）

初期化条件の違い: Faiss k-means（Faiss 既定のランダム初期化）と NumpyEmeans はどちらも学習データから k 点を無作為抽出しますが、乱数生成器が異なるため同一 seed でも初期重心は一致しません（既定 `--init native`）。`--init shared` ではこの2者に同一の初期重心を与えます。これとは別に、より強い Lloyd として **k-means++ 初期化の Faiss k-means（`faiss_kmeans_pp`）** も計測します（初期化時間も学習時間に含む。`--init shared` では除外）。同様に、**それと完全に同じ k-means++ 初期重心から始める e-means（`e_means_pp`）** も計測し、初期化の効果と更新則の効果を切り分けます。詳細は [docs/methodology.md](docs/methodology.md)。

## 5. 評価指標と判定方法

| 指標 | 内容 |
|---|---|
| 学習時間 | `train()` の実時間（秒） |
| train / test MSE | 最終重心に厳密 `faiss.knn` で再割当てした平均二乗距離（全アルゴリズム共通の評価経路） |
| 空クラスタ数 | 学習データの割当てが 0 のクラスタ数 |
| 比較の単位 | (データ, 基準, e-means の種類) ごとに判定。e-means の種類は `e_means`（公式デフォルト初期化）と `e_means_pp`（k-means++ 初期化） |
| 基準 | Faiss k-means 25 反復。仕様上の基準はランダム初期化（`faiss_kmeans@25`、以下 Lloyd25）。k-means++ 版（`faiss_kmeans_pp@25`、以下 Lloyd++25）も別の行として判定 |
| 同等品質到達 | **e-means の test MSE ≤ 基準の test MSE × 1.01** |
| 高速化倍率（vs 基準） | 基準の時間 ÷ 到達した e-means 構成のうち最短の時間 |
| 高速化倍率（vs 同手法最短） | 基準と同じ初期化の k-means が同じ閾値に到達する最短反復数の時間 ÷ 上記 e-means 時間（保守的な倍率） |
| 到達構成の空クラスタ | 到達と判定された e-means 構成の空クラスタ数（MSE だけでは見えない品質問題の確認） |
| シード別到達・倍率 | 各 seed の基準を使って判定した到達数と倍率の中央値・最小・最大 |
| ばらつき | 構成ごとの min / max、時間の変動係数（CV）、test MSE の標準偏差 |

- エポック数の比だけで高速化を判定しません。
- 到達しなかった場合は `emeans_reached = False`（レポートでは **NO**）と記録し、倍率は算出しません。
- 補足として、既存の生データから許容幅だけを変えて（0.5%・0.1%・0%）読み直す `make_tolerance_table.py` があります。判定規則は 1% のままで、補足の読み方です（[docs/tolerance-sensitivity.md](docs/tolerance-sensitivity.md)）。

## 6. 結果ファイルの見方

| 場所 | 内容 | Git |
|---|---|---|
| `results/<run>/raw_results.csv` | 全試行（1 行 = 1 回の学習） | 管理外 |
| `results/<run>/metadata.json` | 実行環境・設定・公式ソースの SHA / ハッシュ | 管理外 |
| `reports/<run>/summary.csv` | 構成（dataset × algorithm × passes）ごとの中央値とばらつき | 管理対象 |
| `reports/<run>/comparison.csv` | (データセット, 基準, e-means の種類) ごとの同等品質到達判定と倍率 | 管理対象 |
| `reports/<run>/metadata.json` | 再現に必要なメタデータ（`results/` と同一） | 管理対象 |
| `reports/<run>/report.md` | 上記をまとめた Markdown レポート | 管理対象 |

`raw_results.csv` の主な列: `dataset, seed, repeat, algorithm, passes, init, run_index, block_position, seconds, cpu_seconds, main_thread_cpu_seconds, loadavg_1m, train_mse, test_mse, empty_clusters, batch_size, alpha0, batch_law, splits`（`batch_size` 以降は e-means のみ）。

計時の診断: `cpu_seconds`（プロセス全体の CPU 時間）、`main_thread_cpu_seconds`（メインスレッドの CPU 時間）、`loadavg_1m`（各実行直前の 1 分ロードアベレージ）を記録し、`report.md` の「計時の診断」節にアルゴリズム別の CPU 時間/経過時間 を示します。経過時間だけが伸びて CPU 時間/経過時間 が下がれば CPU の取り合い（待ち）、比が変わらず全体が伸びればコア自体の遅さ（コア種別・クロック）を示唆します。WSL2 内では CPU クロックは観測できないため記録しません。

## 7. 実測結果（本環境）

実行日: 2026-11-11（速度は `scripts/run_speed_suite.sh` による作業のない時間帯の計測。Medium 19:05、Quick 20:23 JST。どちらも CPU の取り合いの検出 0 件）/ CPU: Intel Core Ultra 7 258V（WSL2、論理コア 6）/ 4 スレッド / faiss-cpu 1.16.0 / Faiss ソース `ddc76806` / 本リポジトリ `5300612`（`repo_dirty: false`）/ 既定 `--init native`。
品質（MSE・空クラスタ）は、日中に計測した `reports/medium_t4/`・`reports/quick_t4_rep3_b/` と全構成で完全に一致しました（同じコードとシードのため）。
以下の数値はすべて `reports/` 配下の実測値から転記したものです。秒・MSE はシード（× 反復）の中央値です。表記: Lloyd25 = ランダム初期化の Faiss k-means 25 反復、Lloyd++25 = k-means++ 初期化の Faiss k-means 25 反復、e-means++ = Lloyd++ と同一の k-means++ 初期重心から始めた e-means。

### Medium（200,000 × 64 次元、k = 1,024、5 シード）— `reports/speed_20261111-1905_medium_t4/`

**A. 仕様上の比較: 公式デフォルトの e-means 対 Lloyd25**

| データ | Lloyd25 の test MSE / 秒 | e-means 到達 | 到達エポック / 秒 | 倍率 vs Lloyd25 | 倍率 vs 同手法最短（反復数） | 到達構成の空クラスタ | e-means 最良 MSE / 基準 | シード別到達 |
|---|---|---|---|---|---|---|---|---|
| gaussian | 54.22 / 6.32 s | yes | 1 / 0.41 s | 15.2× | 3.1×（5） | 0 | 0.999 | 5/5 |
| imbalanced | 114.16 / 6.36 s | **NO** | — | — | —（10） | — | **1.025** | 1/5 |
| separated | 206.94 / 5.11 s | yes | 1 / 0.40 s | 12.7× | 3.6×（5） | **41** | 0.657 | 5/5 |

**B. 初期重心を揃えた比較: e-means++ 対 Lloyd++25（差は更新則のみ。両者とも k-means++ の初期化時間 約 0.9 s を含む）**

| データ | Lloyd++25 の test MSE / 秒 | e-means++ 到達 | 到達エポック / 秒 | 倍率 vs Lloyd++25 | 倍率 vs 同手法最短（反復数） | 到達構成の空クラスタ | e-means++ 最良 MSE / 基準 | シード別到達 |
|---|---|---|---|---|---|---|---|---|
| gaussian | 54.22 / 7.66 s | yes | 1 / 1.32 s | 5.8× | 1.8×（5） | 0 | 0.999 | 5/5 |
| imbalanced | 83.39 / 7.82 s | **NO** | — | — | —（5） | — | **1.239** | 0/5 |
| separated | 121.26 / 6.17 s | yes | 1 / 1.34 s | 4.6× | 1.8×（5） | **16** | 0.866 | 5/5 |

参考（組み合わせを変えた場合）: 公式デフォルトの e-means は Lloyd++25 に対し imbalanced（1.40 倍悪い）・separated（1.12 倍悪い）で未到達。e-means++ は Lloyd25 に対しては全データで到達（imbalanced 5.0×、separated 3.8×）。

e-means++ の separated は、空クラスタが 0 になる 3 エポック（2.21 s, test MSE 111.7）でも、Lloyd++ の最短到達構成（5 反復, 2.38 s, 121.3）より速く良い値でした。

### Quick（20,000 × 64 次元、k = 128、3 シード × 3 反復）— `reports/speed_20261111-2023_quick_t4/`

| データ | 比較 | 到達 | 到達エポック | 倍率 vs 基準 | 倍率 vs 同手法最短 | 最良 MSE / 基準 |
|---|---|---|---|---|---|---|
| gaussian | e-means 対 Lloyd25 | yes | 1 | 4.8× | **0.97×** | 0.999 |
| gaussian | e-means++ 対 Lloyd++25 | yes | 1 | 4.7× | 1.1× | 0.999 |
| imbalanced | e-means 対 Lloyd25 | yes | 2 | 2.9× | **0.61×** | 0.946 |
| imbalanced | e-means++ 対 Lloyd++25 | **NO** | — | — | — | **1.168** |
| separated | e-means 対 Lloyd25 | yes | 1 | 3.0× | 1.0× | 0.505 |
| separated | e-means++ 対 Lloyd++25 | yes | 1 | 2.4× | 1.0× | 0.795 |

**計時の再現性について**: Quick の倍率は、作業のない時間帯の計測でも実行ごとに大きく変わります。gaussian の e-means 対 Lloyd25 は、日中の `reports/quick_t4_rep3_b/` で 6.2×、`reports/quick_t4_rep3/` で 4.0×、夜間の本計測で 4.8× でした。日中の 2 回は e-means の時間だけが約 1.5〜1.6 倍違い、前後のコミットでの A/B 実行では再現しなかったため、コード起因ではありません。夜間の 2 回（19:05 と 20:23、同じコミット）の間でも、e-means 1 エポックの時間の中央値が 1.17 倍違いました。19:05 の回（`reports/speed_20261111-1905_quick_t4/`）は 432 回中 1 回で取り合いを検出したため速度の根拠には使いませんが、同じ比較は 6.2× でした。原因は未特定です（性能コアと省電力コアが混在する CPU でのスレッド配置などの可能性）。品質（MSE）はどの回もビット単位で一致しています。このため **Quick の倍率は実行ごとに 1.5 倍程度変わり得るものとして扱い、時間に関する結論は Medium に基づきます**。

Medium も、日中の計測（`reports/medium_t4/`、作業中で取り合いの影響あり）とは時間が異なります。Lloyd25 は gaussian で 7.68 s → 6.32 s（夜間）と約 18% 短くなった一方、e-means 1 エポックは 0.44 s → 0.41 s と約 5% の短縮にとどまり、倍率は 17.5× → 15.2× に下がりました。日中の値は Lloyd の時間を相対的に大きく見積もっていました。

### 規模を振った結果（n・k・d を 1 つずつ変更、3 シード）— `reports/scale_20261112-0338_*_t4/`

Medium（n=200,000、k=1,024、d=64）を中心に、1 回に 1 つの要因だけを変えました（設計と交絡は [docs/methodology.md](docs/methodology.md) の 6 節）。2026-11-12 03:38〜06:06 JST に `scripts/run_speed_suite.sh` と同じ条件（作業のない時間帯、4 スレッド）で計測し、8 点すべてで CPU の取り合いの検出は 0 件、`repo_dirty: false`（`4ab4c91`）でした。中心の点は上記の Medium（5 シード）を使っています。全データセット・全比較の表は [reports/scale_20261112-0338_overview.md](reports/scale_20261112-0338_overview.md)（`make_scaling_table.py` で生成）にあります。

表の倍率はすべて **vs 同手法最短**（基準と同じ初期化の Lloyd が同じ閾値に届く最短反復数との比）。括弧内は到達した最少エポック（1 以外のとき）と、その構成の空クラスタ数（/ k）。

| 変えた要因 | 値 | 1 エポック / 1 反復の時間比 | e-means のバッチ（1 エポック） | A. e-means 対 Lloyd25: gaussian | B. e-means++ 対 Lloyd++25: gaussian | B: separated | B: imbalanced |
|---|---|---|---|---|---|---|---|
| k | 256 | 2.25 | 4,096 / throughput | 2.23× | 1.53× | 1.49×（空 3） | **NO**（1.09 倍悪い） |
| （中心） | 1,024 | 1.60 | 4,096 / throughput | 3.13× | 1.83× | 1.78×（空 16） | **NO**（1.24） |
| k | 4,096 | 1.22 | 4,096 / throughput | 4.10× | 1.88× | 1.49×（2 ep、空 16） | **NO**（1.21） |
| n | 100,000 | 1.62 | 2,048 / throughput | 3.08× | 1.65× | 1.75×（空 20） | **NO**（1.28） |
| n | 1,000,000 | 1.37 | 16,384 / throughput | 3.64× | 1.67× | 1.56×（空 7） | **NO**（1.14） |
| d | 2 | 23.2 | 512 / quality、復活なし | **0.41×**（3 ep） | **0.79×**（2 ep） | **0.58×**（3 ep） | 1.02×（空 0） |
| d | 16 | 10.7 | 512 / quality | **0.47×** | **0.81×** | **0.71×**（2 ep） | **NO**（1.13） |
| d | 128 | 1.49 | 4,096 / throughput | 3.35× | 1.69× | 1.63×（空 41） | **NO**（1.25） |
| d | 768 | 1.46 | 4,096 / throughput | 3.43× | 1.51× | 1.69×（空 60） | **NO**（1.17） |

時間比は gaussian の e-means 1 エポックの秒 ÷（Lloyd 5 反復の秒 ÷ 5）。バッチは `raw_results.csv` の `batch_size`・`batch_law` 列（d=2 の「復活なし」は `unused_threshold` = 0）。

その他の観察（同じレポートから）:

- **仕様上の比較（e-means 対 Lloyd25、基準の時間 ÷ e-means の時間）** は gaussian の d ≥ 64 で 10.9×（k=256）〜19.6×（k=4,096）。d=16 では 2.3×、d=2 では 0.41×（3 エポック）で、Lloyd25 そのものより遅い。
- **到達と判定された構成の空クラスタが、k と d とともに増える。** e-means++ の 1 エポックで、k=4,096 の separated は 100 / 4,096、d=128 の imbalanced は 123 / 1,024、d=768 の imbalanced は **495 / 1,024（約半数）** が空のまま、test MSE は基準の 1.01 倍以内に入った（Lloyd25 に対する到達）。d=768 の imbalanced は 3 エポックでも 51〜55 個が空。
- **公式デフォルトの e-means 対 Lloyd25 の imbalanced は、点ごとに到達と未到達が入れ替わる。** 中心と n=100,000 では未到達（最良 1.02〜1.03 倍）、k=256・d=128 では 1 エポックで到達、n=1,000,000 では 10 エポックで 2/3 シードのみ到達。基準との差が 1% 前後で判定境界に近いためで、この比較の到達・未到達は条件のわずかな違いで変わる。

### 公開実データ（SIFT1M・DEEP1M、medium 規模）— `reports/speed_20261112-1333_{sift1m,deep1m}_medium_t4/`

1M 行の公開データから 200,000 行を学習、10,000 行を評価に重複なく抽出し、k = 1,024、5 シードで計測しました（データと前処理なしの方針は [docs/methodology.md](docs/methodology.md) の 6b 節）。速度は 2026-11-12 13:33〜13:46 JST に `scripts/run_speed_suite.sh --profiles "" --real "sift1m deep1m"` で計測しました（昼休みで作業なし、VS Code も終了。開始時のロードアベレージ 0.49、2 回とも CPU の取り合いの検出 0 件）。`repo_dirty: false`（`5ebeeb0`）、公式ソースは固定 SHA と一致、入力の SHA-256 も固定値と一致。品質（MSE・空クラスタ）は、同日午前の品質専用の計測（`reports/q_real_{sift1m,deep1m}_medium/`）と全シード・全構成で一致しました。

**A. 仕様上の比較: 公式デフォルトの e-means 対 Lloyd25**

| データ | Lloyd25 の test MSE / 秒 | e-means 到達 | 到達エポック / 秒 | 倍率 vs Lloyd25 | 倍率 vs 同手法最短（反復数） | 到達構成の空クラスタ | e-means 最良 MSE / 基準 | シード別到達（倍率 最小〜最大） |
|---|---|---|---|---|---|---|---|---|
| sift1m（d=128） | 51,576 / 11.45 s | yes | 1 / 0.69 s | 16.5× | 3.1×（5） | 0 | 0.996 | 5/5（13.8〜19.1×） |
| deep1m（d=96） | 0.4702 / 8.54 s | yes | 2 / 1.07 s | 8.0× | 3.5×（10） | 0 | 0.996 | 5/5（7.4〜14.9×） |

**B. 初期重心を揃えた比較: e-means++ 対 Lloyd++25（両者とも k-means++ の初期化時間を含む）**

| データ | Lloyd++25 の test MSE / 秒 | e-means++ 到達 | 到達エポック / 秒 | 倍率 vs Lloyd++25 | 倍率 vs 同手法最短（反復数） | 到達構成の空クラスタ | e-means++ 最良 MSE / 基準 | シード別到達（倍率 最小〜最大） |
|---|---|---|---|---|---|---|---|---|
| sift1m | 51,578 / 12.99 s | yes | 1 / 2.35 s | 5.5× | 2.7×（10） | 0 | 0.996 | 5/5（4.3〜5.7×） |
| deep1m | 0.4709 / 9.92 s | yes | 1 / 1.79 s | 5.5× | 2.8×（10） | 0 | 0.996 | 5/5（4.0〜5.7×） |

参考（組み合わせを変えた場合）: 公式デフォルトの e-means 対 Lloyd++25 は sift1m 18.7×（同手法最短比 9.0×）、deep1m 19.0×（9.7×、1 エポックで到達）。e-means++ 対 Lloyd25 は sift1m 4.9×（**0.93×**、Lloyd 5 反復より遅い）、deep1m 3.5×（1.5×、2 エポック）。

- **人工データの不均衡・分離データで見えた未到達と空クラスタは、どちらの実データでも出なかった。** 全構成・全シードで到達し、空クラスタは 0。振る舞いは人工データの gaussian（クラスタ構造なし）に近い。
- **e-means の 1 エポックは Lloyd の 1 反復の 1.60 倍（sift1m）、1.43 倍（deep1m）**（各データの e-means 1 エポックの秒 ÷（Lloyd 5 反復の秒 ÷ 5））。人工データ medium の 1.4〜1.7 倍と同じ範囲。
- **k-means++ 初期化の時間が e-means++ の大半を占める。** e-means++ と e-means の 1 エポックの差は sift1m 1.65 s、deep1m 1.27 s。sift1m で e-means++ が Lloyd25 の最短到達（5 反復, 2.17 s）より遅いのはこのため。
- **実データでは 1% の許容幅が粗い。** 全手法の test MSE が 1% 程度の幅に収まり、sift1m では Lloyd の 5 反復も基準の 1.009 倍で「到達」になる。deep1m で e-means が Lloyd25 には 2 エポック、Lloyd++25 には 1 エポックで到達するのも、基準の MSE が 0.15% 違うだけで判定が変わるため。判定規則は変えていないが、参考として同じ表の中央値から「基準以下（許容幅 0）」で読むと、e-means / e-means++ はどちらのデータでも 3 エポック（deep1m の e-means 対 Lloyd++25 は 2 エポック）が必要になる。許容幅を変えた再読（[docs/tolerance-sensitivity.md](docs/tolerance-sensitivity.md)）では、0.1% で Lloyd / Lloyd++ も 25 反復を要するため、同手法最短比は下がらず 3.7〜5.9 倍になった（0.5% では e-means++ 対 Lloyd++ だけが約 2.1 倍に下がる）。
- e-means の 10 エポックは、どちらのデータでも Lloyd25 より 0.4〜0.5% 低い test MSE になった（Lloyd は 25 反復でまだ収束していない）。
- deep1m の A は、シード別の倍率が 7.4〜14.9 倍と広い（到達エポックがシードで 1 と 2 に分かれるため）。

### 現時点の解釈（この条件の範囲に限る）

1. **同じ初期重心から比べると、更新則の優劣はデータで逆転する。** e-means++ 対 Lloyd++ では、separated と gaussian は e-means が 1 エポックで同等以上の品質に達し、同手法最短比で約 1.8 倍速い（Medium）。一方 imbalanced では、e-means++ は 10 エポックでも 103.4 で、Lloyd++ の 5 反復（83.7）に届かない。**不均衡データでの劣後は初期化ではなく更新則によるもの**である。さらに品質専用の追加調査（[docs/findings-imbalanced.md](docs/findings-imbalanced.md)）で、**主因は e-means のクラスタ復活（`unused_threshold`、既定 0.1）** と判明した。復活を無効化すると差は 1.24 倍から 1.016 倍に縮む。ただし均等サイズの separated では逆に復活がある方がよく（0.866 対 1.009）、最適な設定はデータ分布に依存する。公式の「追加のハイパーパラメータ不要」という主張に対する反例である。
2. **公式デフォルト（ランダム初期化）の e-means は初期化で損をしている。** 同じ e-means でも k-means++ 初期値にすると Medium の separated は 177→116、imbalanced は 120→108（いずれも 1 エポック）と大きく改善する。ただし k-means++ の初期化には約 0.9 s（Medium gaussian の Lloyd 約 3.6 反復分。e-means++ と e-means の 1 エポックの時間差から推定）かかり、1 エポックの e-means++ の時間の約 7 割を占める。
3. **仕様上の比較（e-means 対 Lloyd25）で見える 13〜15 倍は、比較相手の弱さを含む。** Lloyd は 5〜10 反復で 25 反復と 1% 以内に達しており、同手法最短比では約 3.1〜3.6 倍。さらに Lloyd を k-means++ 初期化にすると、クラスタ構造のあるデータでは公式デフォルトの e-means は品質で届かない。
4. **クラスタ構造のない gaussian では、どの比較でも主張に近い。** 1 エポックで基準と同等の品質。
5. **1 エポックの e-means は空クラスタを残しやすい**（Medium separated: デフォルト 41、e-means++ 16 / 1024）。MSE だけで「到達」と判定される構成でも、実務上は 2〜3 エポック以上が必要な場合がある。
6. **e-means の 1 エポックは Lloyd の 1 反復より重い**（Medium で約 1.4〜1.7 倍、Quick で約 4〜5 倍。Lloyd 5 反復の時間 ÷ 5 との比）。小規模ではエポック数削減の効果が小さく、Quick の gaussian では e-means 1 エポックが Lloyd の最短到達構成（5 反復）とほぼ同じ時間（0.97×）だった（原因は Python 側のオーバーヘッドと推測されるが、未計測）。
7. separated / imbalanced では、いずれの手法も理論上の最良値（約 64、単位分散ノイズ × 64 次元）から遠く、全手法が局所解にある。
8. **規模を振ると、更新則の差（初期重心を揃えた比較）は約 1.5〜1.9 倍でほぼ一定。** d ≥ 64 では k を 16 倍、n を 10 倍、d を 12 倍に変えても、e-means++ 対 Lloyd++ の同手法最短比は gaussian・separated で 1.49〜1.88 倍の範囲に収まり、規模とともに伸びる傾向は見えない。一方、公式デフォルトの e-means 対 Lloyd25 の同手法最短比は k とともに 2.2 → 3.1 → 4.1 倍と伸びた（n・d に対してはほぼ一定）。k が大きいほど 1 エポックの相対コストが下がる（時間比 2.25 → 1.60 → 1.22）ことと対応している。論文の規模（さらに大きい k）での伸びは、本検証では未計測。
9. **低次元（d ≤ 16）では、本環境の CPU 実装の e-means はほぼすべての比較で Lloyd より遅い**（例外は d=2 の imbalanced の e-means++ 対 Lloyd++ で 1.02×）。公式デフォルトが d ≤ 16 で選ぶ quality 則のバッチ（512）では、1 エポックが Lloyd 1 反復の 11〜23 倍かかる。多くの比較で品質には 1〜3 エポックで到達しているので、遅さはエポック数ではなく 1 エポックの重さによる。原因は小さいバッチごとの Python 側の処理と推測されるが、未計測。GPU 実装（本検証の対象外）では事情が異なり得る。
10. **不均衡データでの劣後は、クラスタ復活が有効なすべての点で再現した。** e-means++ 対 Lloyd++ の imbalanced は、復活が有効な 8 点すべてで未到達（1.09〜1.28 倍悪い）。到達したのは、公式デフォルトで復活が無効になる d=2 の点だけ（1 エポック、1.02×）だった。品質専用の追加調査で得た「主因はクラスタ復活」という結論（解釈 1）と整合する。d=2 では quality 則への切り替えも同時に起きるため、品質専用の追加調査（d=2・3 シード、復活の有無 × バッチの決め方の 2 × 2）で切り分けた。復活を無効にするとバッチの決め方によらず到達し（0.976 / 0.978）、復活を有効にするとどちらも未到達だった（1.14 / 1.10）。d=2 で到達したのは復活が無効になったためで、quality 則は主因ではない（[docs/findings-imbalanced.md](docs/findings-imbalanced.md) の追加調査）。
11. **MSE だけの到達判定は、高次元・大きな k で空クラスタを見落とす。** d=768 では、クラスタの約半数が空のまま「到達」と判定される構成があった。判定規則は変えていないが、結果を読むときは `到達構成の空クラスタ` 列を必ず併せて見る必要がある。
12. **公開実データ（SIFT1M・DEEP1M）では、品質・速度とも主張に近い結果が再現した。** 公式デフォルトでも k-means++ 初期化でも、1〜2 エポックで Lloyd 25 反復と 1% 以内の品質に達し、空クラスタも残らなかった。初期重心を揃えた同手法最短比は 2.7〜2.8 倍で、人工データ gaussian（1.8 倍）より大きい。不均衡データで見えた劣後（解釈 1・10）は、この 2 つの実データでは現れなかった。ただし手法間の品質差がもともと 1% 程度と小さく、必要エポックは許容幅に敏感（基準以下まで求めると 2〜3 エポック）で、sift1m の e-means++ 対 Lloyd25 は k-means++ の初期化時間のため 0.93 倍だった。
13. **許容幅を 0.5%・0.1% に狭めて読み直しても、結論が変わるのは 1% で境界付近にあった点だけだった**（[docs/tolerance-sensitivity.md](docs/tolerance-sensitivity.md)、新しい計測なし）。狭めると e-means と Lloyd の両方に追加の反復が要るので、同手法最短比は一方向には動かない（実データでは 0.1% でむしろ上がる）。変わったのは、不均衡データで 10 エポックかけて到達していた 2 点（d=16、n=1M の公式デフォルト）が未到達になること、解釈 9・10 の d=2 の例外（1.02×）が 0.5% で 0.72× になること、実データの e-means++ 対 Lloyd++ が 0.5% で約 2.1 倍に下がること。なお実データと gaussian では基準自体のシード間の幅が 0.04〜0.23% あり、0.1% 以下の判定はこの幅より細かい。

14. **クラスタ復活は、真のクラスタに置かれた重心を引き剥がし、すでに重心のある大きなクラスタへ移す。** 品質専用の診断（[docs/findings-revival.md](docs/findings-revival.md)、公式コードは改変せず、復活の判定ごとに記録）による。d=64 の imbalanced では、復活させられた重心の約 6 割は他の重心と同じクラスタ上にあり、約 4 割はそのクラスタの唯一の重心だった。後者のクラスタは平均の 0.1 倍より大きいことも多いが、判定時の質量は平均の 0.03〜0.06 倍しかなかった（初期の遅れによるものではない。解釈 15）。分割される側は平均の 4〜36 倍の大きなクラスタ上にあった。学習後に重心のない真のクラスタは、復活なしの 266〜453 に対し 352〜711 に増え、MSE の劣後（解釈 1）と対応する。復活の 58〜87% は学習の最初の 4 分の 1 の更新で起きる。均等サイズの separated では、復活させられた重心はすべて重複したもので、重心のないクラスタは逆に減った（復活が効く理由と整合）。

15. **復活の判定を最初の N 回だけ止めても、閾値を下回る重心は減らなかった。標的は「初期に質量が追いついていない」重心ではない。** 品質専用の診断（[docs/findings-revival-warmup.md](docs/findings-revival-warmup.md)）による。判定を止めるのは本リポジトリの改変で、公式のアルゴリズムではない。復活なしで学習の 53〜82% まで進めても、閾値未満の重心は imbalanced で 94〜155（指数 1.0）／ 279〜383（指数 1.3）個のままだった。それでも判定を遅らせるほど imbalanced の劣後は縮み、指数 1.0 で最初の 8 回を止めると復活なしとほぼ同じ MSE になった（10 エポックで 103.5 対 103.1）。指数 1.3 では差が大きく残った（96.5 対 84.7）。後の判定でも、標的の約半分は閾値未満の小さい真のクラスタの唯一の重心で、閾値どおりの劣後と対応する。逆に separated では、遅らせるほど復活の利益が減った（10 エポックで 105.0 → 113.4）。遅らせると効く理由（延べ数・再復活・標的の内訳の変化のどれか）は分けられていない。

## 8. 検証上の制約

- **CPU 上の独立ベンチマークであり、論文の GPU 実験（H100、Flash-KMeans との比較、DEEP / ImageNet-CLIP 等の実データ）の再現ではありません。** 論文の約 10 倍という実時間の数値は、本検証の対象外です。
- 実データは SIFT1M・DEEP1M の 2 種類、medium 規模（n=200,000、k=1,024）のみ。速度は 1 回の計測（昼休みの作業のない時間帯）です。テキスト埋め込みなど他の種類の実データは未検証です。
- Quick 規模の計時は、作業のない時間帯の計測どうしでも e-means の時間が実行間で変わり（日中には約 1.5 倍の事例。原因未特定）、倍率の再現性が低い。
- quick 規模では 1 回の学習が 0.1 秒前後で、Python のオーバーヘッドと計時ノイズの比率が大きくなります。
- WSL2 上の計測であり、ホスト側の負荷や CPU 周波数の変動を受けます。結果は同一マシン内での相対比較として扱ってください。
- e-means の計時には、公式デフォルトによる最終エポックのラベル保存のコストが含まれます。
- シード数が少ないため、min / max は参考値です。
- 規模を振った検証は 1 回に 1 要因だけを変えた 3 シードの計測で、要因間の相互作用（例: 大きな k と大きな n の組み合わせ）は見ていません。k 軸では 1 クラスタあたりの点数も同時に変わります。

## 9. 今後の検証課題

- クラスタ復活: 小さくないクラスタの唯一の重心の質量が小さいままの理由（判定時の割当ての記録）、閾値の細かい掃引、k を変えた再確認（[docs/findings-revival.md](docs/findings-revival.md)・[docs/findings-imbalanced.md](docs/findings-imbalanced.md) の「次の検証候補」）
- 計時ノイズの原因調査（`taskset` 等によるコア固定、Windows の電源設定）。作業のない時間帯でも Quick の時間が実行間で変わる原因の特定
- scikit-learn `MiniBatchKMeans` の追加
- 規模の検証の延長: k = 16,384 以上（1 条件で約 1 時間かかるため未実施）、1 クラスタあたりの点数を一定にした k 軸、k と n を同時に大きくした点
- 低次元で e-means が遅い原因の切り分け（`batch_law="throughput"` を明示した d ≤ 16、1 バッチあたりの処理時間の内訳）
- k や n を変えた実データでの確認
- スレッド数（1, 2, 4, 6）による倍率の変化
- `batch_law="quality"` と `"throughput"` の比較、`capture_assignments=False` の影響
- シード数を増やした（例: 10 以上）統計的な比較
- GPU 環境が用意できた場合の、公式 `benchs/e_means` による論文条件の再現

## ディレクトリ構成

```text
e-means-validation/
├── README.md / CLAUDE.md / LICENSE
├── pyproject.toml / uv.lock / .python-version
├── FAISS_COMMIT              # 検証対象の公式 Faiss コミット SHA
├── benchmark.py              # ベンチマーク CLI
├── make_report.py            # 集計・レポート再生成
├── make_scaling_table.py     # 規模を振った比較表（複数レポートから）
├── make_tolerance_table.py   # 許容幅を変えた再読（生データから再計算、補足）
├── diagnose_revival.py       # クラスタ復活の診断（品質のみ、判定ごとに記録）
├── emv/                      # data / runners / analysis / official / revival
├── prepare_real_data.py      # 公開実データの取得・.npy 変換（SHA-256 固定）
├── scripts/setup_faiss.sh    # 公式ソース取得（SHA 固定）
├── tests/                    # pytest
├── docs/methodology.md       # 検証方法論
├── docs/validation-kit.md    # 既存検証キットからの移行記録
├── docs/tolerance-sensitivity.md  # 許容幅の感度（補足の読み方）
├── docs/findings-*.md        # 品質専用の追加調査（不均衡データ、クラスタ復活、復活の判定の停止）
├── reports/                  # 集計レポート（Git 管理）
├── results/                  # 生データ（Git 管理外）
├── data/                     # 公開実データ（Git 管理外）
└── faiss-src/                # 公式ソース（Git 管理外）
```

GitHub Actions（`.github/workflows/tests.yml`）は軽量なテストのみを実行し、性能比較は行いません。

## ライセンス

本リポジトリのコードは Apache-2.0（`LICENSE`）。e-means を含む Faiss は MIT ライセンスです。e-means を研究に利用する場合は、公式 README の引用情報に従ってください。
