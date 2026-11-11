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

# 品質（MSE）だけを見る実験（他の作業と並行してよい。レポートに時間・倍率を出さない）
uv run python benchmark.py --profile medium --datasets imbalanced --quality-only \
    --emeans-unused-threshold 0 --run-name q_imb_norevival
```

### 速度計測の運用

**速度（時間・倍率）の計測は、他の作業がない時間帯（夜間など）に `scripts/run_speed_suite.sh` で行います。** IDE の操作や他プロセスとの CPU の取り合いで、時間が数十％〜数倍ずれることを実測で確認しているためです（[docs/methodology.md](docs/methodology.md) の「計時の診断」）。

```bash
bash scripts/run_speed_suite.sh                       # quick（3 反復）→ medium、4 スレッド
bash scripts/run_speed_suite.sh --profiles medium --max-wait-min 120
```

- 未コミットの変更があると実行しません（結果をコミットに対応させるため）。
- 1 分ロードアベレージが 1.0 未満になるまで待ってから開始します（`--max-wait-min` 分待っても下がらなければ中止）。
- 各実行で CPU の取り合いが検出された回数を表示し、検出があればレポート冒頭に警告を出します。
- 自動実行する場合の例（Windows タスク スケジューラから WSL を起動）:
  `wsl.exe -d Ubuntu -- bash -lc "cd ~/Work_2404_001/e-means-validation && bash scripts/run_speed_suite.sh"`。
  実行中は Windows がスリープしない設定にしてください。

日中に行う実験は `--quality-only` を付け、品質（MSE・空クラスタ・到達エポック数）だけを評価します。

主なオプション: `--quality-only`、`--emeans-batch-law auto|throughput|quality`、`--emeans-unused-threshold`（e-means のクラスタ復活の閾値。既定は公式デフォルト）、`--imbalance-exponent`（既定 1.3 = キット）、`--kmeans-niters`（既定 5 10 25）、`--kmeans-inits` / `--emeans-inits`（既定 `random kmeanspp`）、`--emeans-epochs`（既定 1 2 3 5 10）、`--baseline-niter`（25）、`--tolerance`（0.01）、`--order shuffled|fixed`、`--no-warmup`、`--input data.npy`（公開データのみ）、`--allow-unpinned`。

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

### 現時点の解釈（この条件の範囲に限る）

1. **同じ初期重心から比べると、更新則の優劣はデータで逆転する。** e-means++ 対 Lloyd++ では、separated と gaussian は e-means が 1 エポックで同等以上の品質に達し、同手法最短比で約 1.8 倍速い（Medium）。一方 imbalanced では、e-means++ は 10 エポックでも 103.4 で、Lloyd++ の 5 反復（83.7）に届かない。**不均衡データでの劣後は初期化ではなく更新則によるもの**である。さらに品質専用の追加調査（[docs/findings-imbalanced.md](docs/findings-imbalanced.md)）で、**主因は e-means のクラスタ復活（`unused_threshold`、既定 0.1）** と判明した。復活を無効化すると差は 1.24 倍から 1.016 倍に縮む。ただし均等サイズの separated では逆に復活がある方がよく（0.866 対 1.009）、最適な設定はデータ分布に依存する。公式の「追加のハイパーパラメータ不要」という主張に対する反例である。
2. **公式デフォルト（ランダム初期化）の e-means は初期化で損をしている。** 同じ e-means でも k-means++ 初期値にすると Medium の separated は 177→116、imbalanced は 120→108（いずれも 1 エポック）と大きく改善する。ただし k-means++ の初期化には約 0.9 s（Medium gaussian の Lloyd 約 3.6 反復分。e-means++ と e-means の 1 エポックの時間差から推定）かかり、1 エポックの e-means++ の時間の約 7 割を占める。
3. **仕様上の比較（e-means 対 Lloyd25）で見える 13〜15 倍は、比較相手の弱さを含む。** Lloyd は 5〜10 反復で 25 反復と 1% 以内に達しており、同手法最短比では約 3.1〜3.6 倍。さらに Lloyd を k-means++ 初期化にすると、クラスタ構造のあるデータでは公式デフォルトの e-means は品質で届かない。
4. **クラスタ構造のない gaussian では、どの比較でも主張に近い。** 1 エポックで基準と同等の品質。
5. **1 エポックの e-means は空クラスタを残しやすい**（Medium separated: デフォルト 41、e-means++ 16 / 1024）。MSE だけで「到達」と判定される構成でも、実務上は 2〜3 エポック以上が必要な場合がある。
6. **e-means の 1 エポックは Lloyd の 1 反復より重い**（Medium で約 1.4〜1.7 倍、Quick で約 4〜5 倍。Lloyd 5 反復の時間 ÷ 5 との比）。小規模ではエポック数削減の効果が小さく、Quick の gaussian では e-means 1 エポックが Lloyd の最短到達構成（5 反復）とほぼ同じ時間（0.97×）だった（原因は Python 側のオーバーヘッドと推測されるが、未計測）。
7. separated / imbalanced では、いずれの手法も理論上の最良値（約 64、単位分散ノイズ × 64 次元）から遠く、全手法が局所解にある。

## 8. 検証上の制約

- **CPU 上の独立ベンチマークであり、論文の GPU 実験（H100、Flash-KMeans との比較、DEEP / ImageNet-CLIP 等の実データ）の再現ではありません。** 論文の約 10 倍という実時間の数値は、本検証の対象外です。
- 人工データ（3 種類、64 次元）のみ。実データ・高次元・低次元は未検証です。
- Quick 規模の計時は、作業のない時間帯の計測どうしでも e-means の時間が実行間で変わり（日中には約 1.5 倍の事例。原因未特定）、倍率の再現性が低い。
- quick 規模では 1 回の学習が 0.1 秒前後で、Python のオーバーヘッドと計時ノイズの比率が大きくなります。
- WSL2 上の計測であり、ホスト側の負荷や CPU 周波数の変動を受けます。結果は同一マシン内での相対比較として扱ってください。
- e-means の計時には、公式デフォルトによる最終エポックのラベル保存のコストが含まれます。
- シード数が少ないため、min / max は参考値です。

## 9. 今後の検証課題

- クラスタ復活のメカニズムの確認（復活の発動回数と対象の質量の記録）、閾値の細かい掃引、k を変えた再確認（[docs/findings-imbalanced.md](docs/findings-imbalanced.md) の「次の検証候補」）
- 計時ノイズの原因調査（`taskset` 等によるコア固定、Windows の電源設定）。作業のない時間帯でも Quick の時間が実行間で変わる原因の特定
- scikit-learn `MiniBatchKMeans` の追加
- クラスタ数（k = 256〜16k）・次元（d = 2, 16, 128, 768）・データ数を振ったスケーリング検証
- 公開実データ（例: SIFT1M、DEEP1M の部分集合）での検証
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
├── emv/                      # data / runners / analysis / official
├── scripts/setup_faiss.sh    # 公式ソース取得（SHA 固定）
├── tests/                    # pytest
├── docs/methodology.md       # 検証方法論
├── docs/validation-kit.md    # 既存検証キットからの移行記録
├── reports/                  # 集計レポート（Git 管理）
├── results/                  # 生データ（Git 管理外）
└── faiss-src/                # 公式ソース（Git 管理外）
```

GitHub Actions（`.github/workflows/tests.yml`）は軽量なテストのみを実行し、性能比較は行いません。

## ライセンス

本リポジトリのコードは Apache-2.0（`LICENSE`）。e-means を含む Faiss は MIT ライセンスです。e-means を研究に利用する場合は、公式 README の引用情報に従ってください。
