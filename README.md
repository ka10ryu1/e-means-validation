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
```

主なオプション: `--kmeans-niters`（既定 5 10 25）、`--kmeans-inits`（既定 `random kmeanspp`）、`--emeans-epochs`（既定 1 2 3 5 10）、`--baseline-niter`（25）、`--tolerance`（0.01）、`--order shuffled|fixed`、`--no-warmup`、`--input data.npy`（公開データのみ）、`--allow-unpinned`。

計測の公平性のために行っていること:

- 同一の学習・評価データ、同一の k・次元数で比較
- `OMP/OPENBLAS/MKL_NUM_THREADS` と `faiss.omp_set_num_threads` でスレッド数を固定
- 計時は `train()` 呼び出しのみ（データ生成・評価は含まない）
- 学習データと評価データ（未学習データ）を分離
- 各データセットで計測外のウォームアップを実施
- 各 (dataset, seed) 内で構成の実行順をシャッフル
- 複数シード（quick 3 / medium 5）で中央値とばらつきを評価
- Faiss k-means の 256 点/クラスタのサンプリング上限を解除（全件学習）

初期化条件の違い: Faiss k-means（Faiss 既定のランダム初期化）と NumpyEmeans はどちらも学習データから k 点を無作為抽出しますが、乱数生成器が異なるため同一 seed でも初期重心は一致しません（既定 `--init native`）。`--init shared` ではこの2者に同一の初期重心を与えます。これとは別に、より強い Lloyd として **k-means++ 初期化の Faiss k-means（`faiss_kmeans_pp`）** も計測します（初期化時間も学習時間に含む。`--init shared` では除外）。詳細は [docs/methodology.md](docs/methodology.md)。

## 5. 評価指標と判定方法

| 指標 | 内容 |
|---|---|
| 学習時間 | `train()` の実時間（秒） |
| train / test MSE | 最終重心に厳密 `faiss.knn` で再割当てした平均二乗距離（全アルゴリズム共通の評価経路） |
| 空クラスタ数 | 学習データの割当てが 0 のクラスタ数 |
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
| `reports/<run>/comparison.csv` | (データセット, 基準) ごとの同等品質到達判定と倍率 | 管理対象 |
| `reports/<run>/metadata.json` | 再現に必要なメタデータ（`results/` と同一） | 管理対象 |
| `reports/<run>/report.md` | 上記をまとめた Markdown レポート | 管理対象 |

`raw_results.csv` の主な列: `dataset, seed, repeat, algorithm, passes, init, run_index, block_position, seconds, train_mse, test_mse, empty_clusters, batch_size, alpha0, batch_law, splits`（`batch_size` 以降は e-means のみ）。

## 7. 実測結果（本環境）

実行日: 2026-11-11 / CPU: Intel Core Ultra 7 258V（WSL2、論理コア 6）/ 4 スレッド / faiss-cpu 1.16.0 / Faiss ソース `ddc76806` / 本リポジトリ `f6ea70e`（`repo_dirty: false`）/ 既定 `--init native`。
以下の数値はすべて `reports/` 配下の実測値から転記したものです。秒・MSE はシード（× 反復）の中央値です。

### Medium（200,000 × 64 次元、k = 1,024、5 シード）— `reports/medium_t4/`

| データ | 基準 | 基準の test MSE / 秒 | e-means 到達 | 到達エポック / 秒 | 倍率 vs 基準 | 倍率 vs 同手法最短（反復数） | 到達構成の空クラスタ | e-means 最良 MSE / 基準 | シード別到達 |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | Lloyd25 | 54.22 / 7.90 s | yes | 1 / 0.51 s | 15.6× | 2.9×（5） | 0 | 0.999 | 5/5 |
| gaussian | Lloyd++25 | 54.22 / 8.90 s | yes | 1 / 0.51 s | 17.5× | 5.3×（5） | 0 | 0.999 | 5/5 |
| imbalanced | Lloyd25 | 114.16 / 7.49 s | **NO** | — | — | —（10） | — | **1.025** | 1/5 |
| imbalanced | Lloyd++25 | 83.39 / 8.89 s | **NO** | — | — | —（5） | — | **1.404** | 0/5 |
| separated | Lloyd25 | 206.94 / 6.14 s | yes | 1 / 0.51 s | 12.0× | 3.2×（5） | **41** | 0.657 | 5/5 |
| separated | Lloyd++25 | 121.26 / 6.59 s | **NO** | — | — | —（5） | — | **1.121** | 0/5 |

### Quick（20,000 × 64 次元、k = 128、3 シード × 3 反復）— `reports/quick_t4_rep3/`

| データ | 基準 | 基準の test MSE / 秒 | e-means 到達 | 到達エポック / 秒 | 倍率 vs 基準 | 倍率 vs 同手法最短（反復数） | 到達構成の空クラスタ | e-means 最良 MSE / 基準 | シード別到達 |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | Lloyd25 | 57.61 / 0.137 s | yes | 1 / 0.021 s | 6.5× | 1.3×（5） | 0 | 0.999 | 3/3 |
| gaussian | Lloyd++25 | 57.62 / 0.147 s | yes | 1 / 0.021 s | 7.0× | 1.7×（5） | 0 | 0.999 | 3/3 |
| imbalanced | Lloyd25 | 142.33 / 0.144 s | yes | 2 / 0.041 s | 3.5× | **0.69×**（5） | 0 | 0.946 | 3/3 |
| imbalanced | Lloyd++25 | 98.04 / 0.149 s | **NO** | — | — | —（5） | — | **1.373** | 0/3 |
| separated | Lloyd25 | 234.98 / 0.068 s | yes | 1 / 0.020 s | 3.4× | 1.2×（5） | 4 | 0.505 | 3/3 |
| separated | Lloyd++25 | 129.75 / 0.076 s | yes | **10** / 0.161 s | **0.47×** | **0.20×**（5） | 0 | 0.914 | 2/3 |

共通初期値（`reports/quick_shared_t4/`、Lloyd25 のみ）でも、Lloyd25 比 3.5〜6.1×、同手法最短比 0.68〜1.6× と傾向は同じでした（imbalanced のシード別到達は 2/3）。quick 規模では同じ構成の中央値時間が実行間で最大約 1.5 倍変動しており、倍率の小数点以下には意味がありません。

### 現時点の解釈（この条件の範囲に限る）

1. **k-means++ 初期化の Lloyd を相手にすると、クラスタ構造のあるデータで e-means は品質面で負ける。** Medium の imbalanced では Lloyd++ の 5 反復（2.5 秒）で test MSE 83.7 に対し、e-means は 10 エポックでも 118.5（最良でも 1.40 倍悪い）。separated でも 1.12 倍悪く、どのシードでも到達しなかった。Quick の separated だけは e-means 10 エポックが Lloyd++ を上回った（MSE 0.91 倍）が、時間は約 2 倍かかった。
2. **ランダム初期化の Lloyd に対する e-means の品質面の優位は、主に Lloyd の初期化の弱さによる。** ランダム初期化の Lloyd は separated / imbalanced で大きく悪い局所解に陥る（Medium separated: 206.9 対 Lloyd++ 121.3）。前回「e-means の最終 MSE が大幅に良い」とした結果は、この比較相手に限った話だった。
3. **クラスタ構造のない gaussian では、主張に近い結果が再現した。** 1 エポックで Lloyd25 / Lloyd++25 と同等の品質に達し、倍率は vs 基準 15.6〜17.5×、vs 同手法最短 2.9〜5.3×（Medium）。
4. **「25 反復比」の倍率は過大に見えやすい。** Faiss k-means は全条件で 5〜10 反復で 25 反復と 1% 以内の test MSE に達しており、それと比べると倍率は数分の一に縮む。
5. **e-means の 1 エポックは Lloyd の 1 反復より重い**（gaussian で比較すると Medium で約 1.6 倍、Quick で約 4 倍）。小規模ではエポック数削減の効果が小さい（原因は Python 側のオーバーヘッドと推測されるが、未計測）。
6. **1 エポックの e-means は空クラスタを残しやすい**（Medium separated で 41/1024、imbalanced で 22）。MSE だけで「到達」と判定される構成でも、実務上は 2〜3 エポック以上が必要な場合がある。
7. separated / imbalanced では、いずれの手法も理論上の最良値（約 64、単位分散ノイズ × 64 次元）から遠く、3 手法とも局所解にある。

## 8. 検証上の制約

- **CPU 上の独立ベンチマークであり、論文の GPU 実験（H100、Flash-KMeans との比較、DEEP / ImageNet-CLIP 等の実データ）の再現ではありません。** 論文の約 10 倍という実時間の数値は、本検証の対象外です。
- 人工データ（3 種類、64 次元）のみ。実データ・高次元・低次元は未検証です。
- e-means は公式デフォルトのランダム初期化のみで計測しています。e-means に k-means++ の初期値を与えた場合は未検証です。
- quick 規模では 1 回の学習が 0.1 秒前後で、Python のオーバーヘッドと計時ノイズの比率が大きくなります。
- WSL2 上の計測であり、ホスト側の負荷や CPU 周波数の変動を受けます。結果は同一マシン内での相対比較として扱ってください。
- e-means の計時には、公式デフォルトによる最終エポックのラベル保存のコストが含まれます。
- シード数が少ないため、min / max は参考値です。

## 9. 今後の検証課題

- e-means に k-means++ の初期値を与えた比較（`init_centroids`）、scikit-learn `MiniBatchKMeans` の追加
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
