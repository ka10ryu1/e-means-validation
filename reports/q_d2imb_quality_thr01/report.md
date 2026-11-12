# e-means 検証レポート: q_d2imb_quality_thr01

> 本レポートは `benchmark.py` の実測値のみから自動生成されています。CPU上の独立ベンチマークであり、論文のGPU実験（H100, Flash-KMeans比較）の再現ではありません。

> **品質専用の計測**: 他の作業と並行して実行しており、時間は速度の評価に使わない。表からも時間・倍率の列を除いている（生データには記録あり）。

## 実行条件

- プロファイル: `medium` (ntrain=200000, ntest=10000, dim=2, k=1024)  — プロファイルの値を上書き: dim=2
- データ: imbalanced  / 入力: synthetic
- 乱数シード: [0, 1, 2]  / 反復: 1  / 実行順: shuffled
- 初期化: `native` — Faiss k-means と e-means はそれぞれ独自の乱数でk点を学習データから抽出（同一seedでも初期重心は一致しない）
- スレッド数: 4 (CPU: Intel(R) Core(TM) Ultra 7 258V, 論理コア 6)
- Faiss k-means 反復数: [5, 10, 25]  / e-means エポック数: [1, 2, 3, 5, 10]
- e-means 設定: batch_law=`quality`, unused_threshold=`0.1`（None = 公式デフォルト）  / imbalanced の指数: 1.3
- faiss-cpu 1.16.0, numpy 2.4.6, Python 3.11.17
- 公式ソース: facebookresearch/faiss @ `ddc768060c42e9627b6bf22123fe7e207c9a6efc` (固定SHAと一致: True, e_means *.py sha256 `ed83831f538d36d9…`)
- 実行日時: 2026-11-12T06:32:50+00:00  / 本リポジトリ commit: `d60528f118bd80c5c3a962f6d27ad87bb2368173`

## 同等品質到達の判定

判定規則: test MSE <= 基準（Faiss k-means 25反復。ランダム初期化、計測時は k-means++ 初期化も）の test MSE × 1.01（中央値どうし／シード別の両方で判定）。到達しなかった場合は **NO** と記録する。
「最少到達エポック」は閾値に達した最も少ない e-means エポック数（時間に依存しない）。
基準 `faiss_kmeans` は Faiss 既定のランダム初期化、`faiss_kmeans_pp` は k-means++ 初期化（初期化時間も学習時間に含む）。`e_means` は公式デフォルトの初期化、`e_means_pp` は `faiss_kmeans_pp` と同一の k-means++ 初期重心から開始（初期化時間を含む）。以下の列名の emeans_* は、この行の e-means 初期化（candidate）を指す。

| dataset | 基準 | e-means 初期化 | 基準 test MSE | e-means 到達 | 最少到達エポック | その空クラスタ | 同手法の最短到達反復 | e-means最良MSE / 基準 | シード別到達 |
|---|---|---|---|---|---|---|---|---|---|
| imbalanced | faiss_kmeans@25 | e_means | 0.058661 | yes | 3 | 0 | 25 | 0.91814 | 3/3 |
| imbalanced | faiss_kmeans@25 | e_means_pp | 0.058661 | yes | 1 | 0 | 25 | 0.76518 | 3/3 |
| imbalanced | faiss_kmeans_pp@25 | e_means | 0.039445 | **NO** |  |  | 25 | 1.3654 | 0/3 |
| imbalanced | faiss_kmeans_pp@25 | e_means_pp | 0.039445 | **NO** |  |  | 25 | 1.1379 | 0/3 |

## 構成別の集計（シード×反復の中央値とばらつき）

| dataset | algorithm | passes | runs | train MSE | test MSE | test min | test max | 空クラスタ 中央値 | 空 max |
|---|---|---|---|---|---|---|---|---|---|
| imbalanced | e_means | 1 | 3 | 0.060326 | 0.068127 | 0.065574 | 0.069514 | 0 | 0 |
| imbalanced | e_means | 2 | 3 | 0.053432 | 0.059808 | 0.059448 | 0.064273 | 0 | 0 |
| imbalanced | e_means | 3 | 3 | 0.051056 | 0.058624 | 0.05645 | 0.059742 | 0 | 0 |
| imbalanced | e_means | 5 | 3 | 0.049592 | 0.056279 | 0.055817 | 0.057009 | 0 | 0 |
| imbalanced | e_means | 10 | 3 | 0.047009 | 0.053859 | 0.052372 | 0.055267 | 0 | 0 |
| imbalanced | e_means_pp | 1 | 3 | 0.04758 | 0.053679 | 0.05269 | 0.054023 | 0 | 0 |
| imbalanced | e_means_pp | 2 | 3 | 0.040546 | 0.04723 | 0.046527 | 0.04926 | 0 | 0 |
| imbalanced | e_means_pp | 3 | 3 | 0.041235 | 0.047566 | 0.045668 | 0.048485 | 0 | 0 |
| imbalanced | e_means_pp | 5 | 3 | 0.039352 | 0.045763 | 0.045761 | 0.046619 | 0 | 0 |
| imbalanced | e_means_pp | 10 | 3 | 0.03878 | 0.044886 | 0.043981 | 0.046364 | 0 | 0 |
| imbalanced | faiss_kmeans | 5 | 3 | 0.065081 | 0.074398 | 0.073505 | 0.076233 | 0 | 0 |
| imbalanced | faiss_kmeans | 10 | 3 | 0.058478 | 0.066566 | 0.065061 | 0.067 | 0 | 0 |
| imbalanced | faiss_kmeans | 25 | 3 | 0.051742 | 0.058661 | 0.057729 | 0.06074 | 0 | 0 |
| imbalanced | faiss_kmeans_pp | 5 | 3 | 0.036942 | 0.040502 | 0.040249 | 0.040726 | 0 | 0 |
| imbalanced | faiss_kmeans_pp | 10 | 3 | 0.035874 | 0.039885 | 0.039763 | 0.040041 | 0 | 0 |
| imbalanced | faiss_kmeans_pp | 25 | 3 | 0.035188 | 0.039445 | 0.039397 | 0.039564 | 0 | 0 |

## 計時の診断

CPU 時間 / 経過時間 の中央値（プロセス全体・メインスレッド）と、各実行直前の 1 分ロードアベレージ（本プロセス自身のスレッドを含む）。他の回と比べて経過時間だけが伸び、CPU 時間/経過時間 が下がっていればCPU の取り合い（待ち）、比が変わらず時間全体が伸びていればコア自体の遅さ（コア種別・クロック）を示唆する。WSL2 内では CPU クロックは観測できない。「取り合い検出」= メインスレッドの CPU 時間が経過時間の 95% 未満だった実行（学習スレッドは I/O や sleep をしないため、他プロセスに CPU を奪われていたことを示す）。

- ロードアベレージ（1/5/15 分） 開始時: [2.78, 1.11, 1.02]  / 終了時: [3.52, 1.58, 1.19]

| algorithm | runs | プロセスCPU/経過 | メインスレッドCPU/経過 | load 1m 中央値 | load 1m 最大 | 取り合い検出数 |
|---|---|---|---|---|---|---|
| e_means | 15 | 3.997 | 1.001 | 3.27 | 3.52 | 0 |
| e_means_pp | 15 | 3.976 | 0.994 | 3.27 | 3.48 | 0 |
| faiss_kmeans | 9 | 3.99 | 1 | 3.12 | 3.48 | 0 |
| faiss_kmeans_pp | 9 | 3.975 | 0.996 | 3.12 | 3.48 | 0 |

## 注意事項

- MSE は全アルゴリズム共通の厳密 `faiss.knn`（二乗L2）で最終重心に再割当てして算出。e-means の train() が返すラベルは使用していない。
- シード数が少ない場合、中央値・min/max は参考値として扱うこと。
