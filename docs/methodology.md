# 検証方法論

## 1. 検証対象の主張

公式README（`faiss/contrib/e_means/README.md`）および論文（arXiv:2611.09501）の主張:

- e-means は Lloyd の k-means に対し **エポック数・実時間とも約10倍高速**で、かつ **損失が低い**。
- 追加のハイパーパラメータは不要（バッチサイズ・初期ステップ幅はスケーリング則で自動決定）。
- 5エポックで通常 Lloyd の収束時損失を上回る。

本リポジトリはこれを **前提とせず**、CPU上で次を実測する。

1. e-means は従来の k-means より本当に高速か
2. 同等品質に、より少ないエポック数で到達するか
3. データ分布・次元・クラスタ数で結果が変わるか
4. 実務導入の価値があるか

論文の主要な実時間比較は H100 GPU 上で Flash-KMeans を相手に行われており、本検証（CPU, Faiss k-means 相手）はその **再現ではない**。

## 2. 比較対象

| 名称 | 実装 | 設定 |
|---|---|---|
| `faiss_kmeans` | `faiss.Kmeans`（C++ Lloyd法） | `niter` ∈ {5, 10, 25}、`max_points_per_centroid=n`（既定の 256点/クラスタ サンプリング上限を解除し全件学習）、`min_points_per_centroid=1`、`nredo=1`、初期化 RANDOM |
| `faiss_kmeans_pp` | `faiss.Kmeans`（C++ Lloyd法） | `faiss_kmeans` と同じ設定で、初期化のみ k-means++（`init_method=ClusteringInitMethod_KMEANS_PLUS_PLUS`）。初期化の計算時間も `train()` 内で計時される |
| `e_means_pp` | 公式 `NumpyEmeans`（同上） | `e_means` と同じ設定で、`faiss_kmeans_pp` と**ビット単位で同一の** k-means++ 初期重心（`faiss.ClusteringInitialization`, seed+1）から開始。初期化時間も計時に含む |
| `e_means` | 公式 `faiss.contrib.e_means.NumpyEmeans`（FAISS_COMMIT 固定） | `nepochs` ∈ {1, 2, 3, 5, 10}、その他すべて公式デフォルト（`batch_law="auto"`, `capture_assignments=True`, 縮退クラスタの復活あり） |

次フェーズ候補: scikit-learn `MiniBatchKMeans`。

`--kmeans-inits` / `--emeans-inits` で初期化方式を選ぶ（既定はどちらも `random kmeanspp`）。`faiss_kmeans_pp` と `e_means_pp` は初期重心が完全に一致するため、両者の差は更新則（Lloyd 対 e-means）だけによる。仕様上の基準はランダム初期化（Faiss 既定）の Lloyd25 で、k-means++ 版は「より強い Lloyd」として別の基準行で比較する。

### 初期化条件の違い

| | Faiss k-means（random） | Faiss k-means（k-means++） | NumpyEmeans |
|---|---|---|---|
| 方式 | 学習データから k 点を一様無作為抽出（`rand_perm`, seed+1） | D(x)² に比例する確率で逐次選択 | 学習データから k 点を一様無作為抽出（`np.random.RandomState(seed).permutation`） |
| 同一 seed での初期重心 | **3者とも一致しない** | | |
| 初期化コスト | ほぼ0 | O(n·k·d)（計時に含む） | ほぼ0 |
| 空クラスタ処理 | 各反復で大きなクラスタを分割して補充 | 同左 | 5更新ごとに質量が平均の0.1倍未満のクラスタを、大きいクラスタの分割で復活（d≤2 では無効） |

`--init shared` を指定すると、random の k-means と e-means に `RandomState(seed).permutation(n)[:k]` の同一初期重心を与える（e-means にとってはデフォルトと同一の初期値）。初期値を外から与えるため、このモードでは k-means++ 系（`faiss_kmeans_pp`, `e_means_pp`）は自動的に除外される。既定は `native`（各ライブラリの通常の使い方）。

## 3. データ

`emv/data.py`（キットから無変更）。データ生成用乱数は `20261111` 固定で、学習・評価は同一分布からの独立な標本。

| 名前 | 生成 |
|---|---|
| `separated` | k 個の中心 ~ N(0, 4²)、各点 = 中心 + N(0, 1)、クラスタサイズ均等 |
| `gaussian` | 単一の標準正規分布（クラスタ構造なし） |
| `imbalanced` | `separated` と同じ中心、クラスタサイズ ∝ rank^-1.3 |

| プロファイル | ntrain | ntest | dim | k | 既定シード |
|---|---|---|---|---|---|
| smoke（テスト専用） | 3,000 | 1,000 | 16 | 16 | 0 1 |
| quick | 20,000 | 4,000 | 64 | 128 | 0 1 2 |
| medium | 200,000 | 10,000 | 64 | 1,024 | 0 1 2 3 4 |

`separated` / `imbalanced` の理論上の最良 MSE は概ね次元数（=64, 単位分散ノイズ）付近であり、それより大きい値は局所解に陥っていることを示す。

## 4. 計測手順

1. `--threads N` を `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` に設定してから numpy・faiss を import し、`faiss.omp_set_num_threads(N)` も呼ぶ。
2. 各データセットで、計測外のウォームアップ（両アルゴリズムを最大 50k 行で1エポック）を行う。
3. 各 (dataset, seed, repeat) ブロック内で全構成の実行順をシャッフル（`--order fixed` で無効化可）。`run_index`・`block_position` をCSVに記録。
4. 計時は `train()` 呼び出しのみ（`time.perf_counter`）。データ生成・評価は含まない。
5. 計時の診断として、各実行の CPU 時間（プロセス全体 `process_time`、メインスレッド `thread_time`）と直前の 1 分ロードアベレージを記録する。計時異常が起きたとき、待ち（取り合い）とコアの遅さを区別するため。計測自体への影響は時計の読み取り 6 回のみ。
   メインスレッドの CPU 時間が経過時間の 95% 未満の実行を「CPU の取り合いあり」と判定し、レポート冒頭に警告を出す（学習スレッドは I/O・sleep をしないため、下回るのは他プロセスに CPU を奪われた場合）。2026-11-11 の確認では、VS Code の拡張機能ホストと Pylance の起動が重なった区間で比率が 0.84 に下がり、経過時間が 5 倍になった。
6. 速度の計測は他の作業がない時間帯に `scripts/run_speed_suite.sh`（規模を振る検証は `scripts/run_scaling_suite.sh`）で行う。日中の実験は `--quality-only` とし、時間を評価に使わない。
7. 評価は全構成共通に、最終重心へ厳密 `faiss.knn`（二乗L2）で再割当てして MSE と空クラスタ数を算出。

## 5. 評価指標と判定方法

| 指標 | 定義 |
|---|---|
| `seconds` | 学習時間（秒） |
| `train_mse` / `test_mse` | 学習・評価データの最近傍重心への平均二乗距離 |
| `empty_clusters` | 学習データを最終重心に割り当てたとき点が0のクラスタ数 |
| 基準 | `faiss_kmeans@25`（仕様上の基準、ランダム初期化）と、計測していれば `faiss_kmeans_pp@25`。`comparison.csv` は (dataset, 基準, e-means の種類 `candidate`) ごとに1行 |
| 同等品質到達 | `test_mse ≤ 基準の test_mse × 1.01`（1%以内、または良い） |
| `speedup_vs_baseline` | 基準の中央値時間 ÷ 到達した e-means 構成のうち最短の中央値時間 |
| `speedup_vs_fastest_kmeans` | 同じ閾値に到達する **同じ初期化方式の k-means の最短反復数** の時間 ÷ 上記 e-means 時間（k-means が25反復より前に収束している場合の過大評価を防ぐ保守的な倍率） |
| `emeans_best_median_empty_clusters` | 到達と判定された e-means 構成の空クラスタ数の中央値（MSE 基準だけでは見えない品質問題の確認用） |
| `per_seed_*` | 各 seed 内で、その seed の基準を使って判定した到達数と倍率の中央値・最小・最大 |
| ばらつき | 構成ごとの min/max、時間の変動係数（CV）、test MSE の標準偏差 |

- 中央値どうしの判定はキットの規則を踏襲。シード別判定はばらつきを示すための補助。
- e-means が閾値に到達しなかった場合、倍率は空欄とし `emeans_reached = False`（レポートでは **NO**）と記録する。
- エポック数の比だけで高速化を判定しない。
- 許容幅の感度: `make_tolerance_table.py` は既存の `raw_results.csv` から同じ `compare()` で許容幅だけを変えて再計算する（既定 1% / 0.5% / 0.1% / 0%）。判定規則は 1% のままで、補足の読み方として [tolerance-sensitivity.md](tolerance-sensitivity.md) に記録する。記録された許容幅での再計算が `comparison.csv` と一致しなければエラーにする。倍率は時間を計測し、CPU の取り合いを検出しなかった実行のみ表示する。

## 6. 規模を振る検証（スケーリング）

目的: Medium で観察した倍率（同手法最短比で約 3 倍）や不到達が、データ数 n・クラスタ数 k・次元 d を変えたときに伸びるか、縮むか、逆転するかを確かめる。

設計: Medium（n=200,000、k=1,024、d=64）を中心に、**1 回に 1 つの要因だけ**を変える（`scripts/run_scaling_suite.sh`）。

| 軸 | 値 | 固定するもの |
|---|---|---|
| k | 256、（1,024）、4,096 | n=200,000、d=64 |
| n | 100,000、（200,000）、1,000,000 | k=1,024、d=64 |
| d | 2、16、（64）、128、768 | n=200,000、k=1,024 |

- 括弧内は Medium の点で、再計測せず最新の夜間の Medium レポートを使う（Medium は 5 シード、他の点は 3 シード）。
- データ生成・判定規則・計時・評価は Medium と同一。`--ntrain` / `--ntest` / `--dim` / `--k` でプロファイルの値を上書きするだけで、生成ロジックは変えていない（上書きした値は `metadata.json` の `settings_overrides` とレポートの「実行条件」に記録）。
- 速度は作業のない時間帯にのみ計測する（4 節の 6）。

この設計に含まれる交絡（結果の解釈で考慮する）:

- **k 軸では 1 クラスタあたりの点数も変わる**（k=256 で約 780 点、4,096 で約 49 点）。点数を一定にした k 軸（n を k に比例させる）は計算量が大きく、今回は対象外。
- **公式のハイパーパラメータ則の当てはめ範囲**: 公式ソース（`contrib/e_means/base.py` のコメント）によると、バッチ則とステップ則は n = 1e5〜1e6（一部 1e7、1e8）、K = 10〜1e4 の範囲で当てはめられている。n=100,000〜1,000,000、k ≤ 4,096 は範囲内。quick（n=20,000）は範囲外である。
- **d によって e-means の設定自体が変わる**: 公式デフォルト `batch_law="auto"` は d ≤ 16 で quality 則（バッチが throughput 則の約 1/8）を選び、`unused_threshold=None` は d ≤ 2 でクラスタ復活を無効にする。d 軸の変化には、次元そのものの効果とこれらの切り替えの効果が混ざる。実際に使われた値は `raw_results.csv` の `batch_size`・`batch_law`・`unused_threshold` 列で確認できる。
- **人工データの分離度**: 中心は各次元 N(0, 4²)、ノイズは単位分散なので、中心間距離とノイズの比は d によらない。ただし d=2 では k=1,024 個のブロブが平面上で重なり合い、`separated` も実質的に分離していない。

## 6b. 公開実データ

目的: 人工データ（ガウスブロブ）で得た結論が、生成分布の分かっていない実際の特徴ベクトルでも成り立つかを確かめる。

| 名前 | 内容 | 行 × 次元 | 取得元・ライセンス |
|---|---|---|---|
| `sift1m` | SIFT 局所特徴（TEXMEX `sift_base.fvecs`）。非負で整数値、未正規化 | 1,000,000 × 128 | INRIA TEXMEX corpus（`ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`） |
| `deep1m` | CNN 画像埋め込み（DEEP1B `base.1B.fbin` の先頭 1,000,000 行。big-ann-benchmarks の Deep1M と同じ切り出し） | 1,000,000 × 96 | Yandex、CC BY 4.0（HTTP の範囲取得で先頭だけを取得） |

- 取得と変換は `prepare_real_data.py`。配布されたままの float32 値を使い、正規化・PCA などの前処理はしない。ダウンロードと `.npy` の SHA-256 をスクリプト内に固定し、不一致はエラーにする。`.npy` の SHA-256 は各実行の `metadata.json` の `input_sha256` にも記録する。
- 規模は medium と同じ（`--profile medium --input data/<name>.npy`）: 1M 行から重複なく 200,000 行を学習、10,000 行を評価に無作為抽出（`DATA_SEED` 固定。`emv/data.py` の `real_embeddings`）。k=1,024、5 シード。判定規則・計時・評価は人工データと同一。
- レポートのデータセット名は入力ファイル名（`sift1m` / `deep1m`）。

交絡・制約:

- **次元が medium（64）と違う**（128、96）。人工データの medium との差には、データの性質と次元の差が混ざる（d=128 の人工データは規模を振った検証の `scale_*_d128_t4` にある）。
- **理論上の最良 MSE が分からない**。人工データのような「MSE ≈ 次元数なら正解」という目安はなく、基準（Lloyd 25 反復）との相対比較だけで判断する。
- 1M 行のうち 21 万行しか使わない。全行での学習は対象外。
- DEEP1M には完全に同一の行が少数ある（先頭 20 万行で 26 行）。除去せずそのまま使う。

## 7. 既知の制約・交絡要因

- **規模依存**: quick（20k×128）は1回の学習が数十〜百数十ミリ秒で、Python のループ・関数呼び出しのオーバーヘッドの比率が大きい。倍率は規模に強く依存し得る。
- **Lloyd の初期化**: ランダム初期化の Lloyd は分離の良いデータで局所解に陥りやすい。このため k-means++ 版を2つ目の基準として併記している。e-means 側も k-means++ 初期値から始める `e_means_pp` を併記している。
- **e-means の計時にはラベル保存コストを含む**（公式デフォルト）。
- **WSL2 上の計測**: ホストOSの負荷、CPU周波数変動、電源設定の影響を受ける。同一マシン内の相対比較に限定すること。
- **共有CIランナーでは性能を評価しない**（GitHub Actions は正しさのテストのみ）。
