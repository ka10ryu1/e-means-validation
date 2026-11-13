# e-means 検証レポート: q_imb_exp10_thr001

> 本レポートは `benchmark.py` の実測値のみから自動生成されています。CPU上の独立ベンチマークであり、論文のGPU実験（H100, Flash-KMeans比較）の再現ではありません。

> **品質専用の計測**: 他の作業と並行して実行しており、時間は速度の評価に使わない。表からも時間・倍率の列を除いている（生データには記録あり）。

## 実行条件

- プロファイル: `medium` (ntrain=200000, ntest=10000, dim=64, k=1024)
- データ: imbalanced  / 入力: synthetic
- 乱数シード: [0, 1, 2, 3, 4]  / 反復: 1  / 実行順: shuffled
- 初期化: `native` — Faiss k-means と e-means はそれぞれ独自の乱数でk点を学習データから抽出（同一seedでも初期重心は一致しない）
- スレッド数: 4 (CPU: Intel(R) Core(TM) Ultra 7 258V, 論理コア 6)
- Faiss k-means 反復数: [25]  / e-means エポック数: [1, 2, 3, 5, 10]
- e-means 設定: batch_law=`auto`, unused_threshold=`0.01`（None = 公式デフォルト）  / imbalanced の指数: 1.0
- faiss-cpu 1.16.0, numpy 2.4.6, Python 3.11.17
- 公式ソース: facebookresearch/faiss @ `ddc768060c42e9627b6bf22123fe7e207c9a6efc` (固定SHAと一致: True, e_means *.py sha256 `ed83831f538d36d9…`)
- 実行日時: 2026-11-13T01:49:29+00:00  / 本リポジトリ commit: `2fd6270de10102580cf2af780e089bc2e59ff543`

## 同等品質到達の判定

判定規則: test MSE <= 基準（Faiss k-means 25反復。ランダム初期化、計測時は k-means++ 初期化も）の test MSE × 1.01（中央値どうし／シード別の両方で判定）。到達しなかった場合は **NO** と記録する。
「最少到達エポック」は閾値に達した最も少ない e-means エポック数（時間に依存しない）。
基準 `faiss_kmeans` は Faiss 既定のランダム初期化、`faiss_kmeans_pp` は k-means++ 初期化（初期化時間も学習時間に含む）。`e_means` は公式デフォルトの初期化、`e_means_pp` は `faiss_kmeans_pp` と同一の k-means++ 初期重心から開始（初期化時間を含む）。以下の列名の emeans_* は、この行の e-means 初期化（candidate）を指す。

| dataset | 基準 | e-means 初期化 | 基準 test MSE | e-means 到達 | 最少到達エポック | その空クラスタ | 同手法の最短到達反復 | e-means最良MSE / 基準 | シード別到達 |
|---|---|---|---|---|---|---|---|---|---|
| imbalanced | faiss_kmeans@25 | e_means_pp | 163.87 | yes | 1 | 10 | 25 | 0.61153 | 5/5 |
| imbalanced | faiss_kmeans_pp@25 | e_means_pp | 103.05 | yes | 2 | 0 | 25 | 0.97247 | 5/5 |

## 構成別の集計（シード×反復の中央値とばらつき）

| dataset | algorithm | passes | runs | train MSE | test MSE | test min | test max | 空クラスタ 中央値 | 空 max |
|---|---|---|---|---|---|---|---|---|---|
| imbalanced | e_means_pp | 1 | 5 | 105.86 | 107.1 | 105.78 | 109.37 | 10 | 12 |
| imbalanced | e_means_pp | 2 | 5 | 100.98 | 102.92 | 102.09 | 103.59 | 0 | 2 |
| imbalanced | e_means_pp | 3 | 5 | 100.55 | 102.66 | 101.56 | 103.7 | 0 | 0 |
| imbalanced | e_means_pp | 5 | 5 | 98.018 | 100.21 | 99.145 | 102.04 | 0 | 0 |
| imbalanced | e_means_pp | 10 | 5 | 98.082 | 100.35 | 98.776 | 101.52 | 0 | 0 |
| imbalanced | faiss_kmeans | 25 | 5 | 163.64 | 163.87 | 158.77 | 167.92 | 0 | 0 |
| imbalanced | faiss_kmeans_pp | 25 | 5 | 100.94 | 103.05 | 100.83 | 104.26 | 0 | 0 |

## 計時の診断

CPU 時間 / 経過時間 の中央値（プロセス全体・メインスレッド）と、各実行直前の 1 分ロードアベレージ（本プロセス自身のスレッドを含む）。他の回と比べて経過時間だけが伸び、CPU 時間/経過時間 が下がっていればCPU の取り合い（待ち）、比が変わらず時間全体が伸びていればコア自体の遅さ（コア種別・クロック）を示唆する。WSL2 内では CPU クロックは観測できない。「取り合い検出」= メインスレッドの CPU 時間が経過時間の 95% 未満だった実行（学習スレッドは I/O や sleep をしないため、他プロセスに CPU を奪われていたことを示す）。

- ロードアベレージ（1/5/15 分） 開始時: [4.37, 4.7, 4.05]  / 終了時: [4.27, 4.51, 4.08]

| algorithm | runs | プロセスCPU/経過 | メインスレッドCPU/経過 | load 1m 中央値 | load 1m 最大 | 取り合い検出数 |
|---|---|---|---|---|---|---|
| e_means_pp | 25 | 3.922 | 1 | 4.26 | 4.44 | 0 |
| faiss_kmeans | 5 | 3.981 | 0.997 | 4.24 | 4.34 | 0 |
| faiss_kmeans_pp | 5 | 3.983 | 0.998 | 4.29 | 4.44 | 0 |

## 注意事項

- MSE は全アルゴリズム共通の厳密 `faiss.knn`（二乗L2）で最終重心に再割当てして算出。e-means の train() が返すラベルは使用していない。
- シード数が少ない場合、中央値・min/max は参考値として扱うこと。
