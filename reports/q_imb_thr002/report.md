# e-means 検証レポート: q_imb_thr002

> 本レポートは `benchmark.py` の実測値のみから自動生成されています。CPU上の独立ベンチマークであり、論文のGPU実験（H100, Flash-KMeans比較）の再現ではありません。

> **品質専用の計測**: 他の作業と並行して実行しており、時間は速度の評価に使わない。表からも時間・倍率の列を除いている（生データには記録あり）。

## 実行条件

- プロファイル: `medium` (ntrain=200000, ntest=10000, dim=64, k=1024)
- データ: imbalanced  / 入力: synthetic
- 乱数シード: [0, 1, 2, 3, 4]  / 反復: 1  / 実行順: shuffled
- 初期化: `native` — Faiss k-means と e-means はそれぞれ独自の乱数でk点を学習データから抽出（同一seedでも初期重心は一致しない）
- スレッド数: 4 (CPU: Intel(R) Core(TM) Ultra 7 258V, 論理コア 6)
- Faiss k-means 反復数: [25]  / e-means エポック数: [1, 2, 3, 5, 10]
- e-means 設定: batch_law=`auto`, unused_threshold=`0.02`（None = 公式デフォルト）  / imbalanced の指数: 1.3
- faiss-cpu 1.16.0, numpy 2.4.6, Python 3.11.17
- 公式ソース: facebookresearch/faiss @ `ddc768060c42e9627b6bf22123fe7e207c9a6efc` (固定SHAと一致: True, e_means *.py sha256 `ed83831f538d36d9…`)
- 実行日時: 2026-11-13T01:34:02+00:00  / 本リポジトリ commit: `2fd6270de10102580cf2af780e089bc2e59ff543`

## 同等品質到達の判定

判定規則: test MSE <= 基準（Faiss k-means 25反復。ランダム初期化、計測時は k-means++ 初期化も）の test MSE × 1.01（中央値どうし／シード別の両方で判定）。到達しなかった場合は **NO** と記録する。
「最少到達エポック」は閾値に達した最も少ない e-means エポック数（時間に依存しない）。
基準 `faiss_kmeans` は Faiss 既定のランダム初期化、`faiss_kmeans_pp` は k-means++ 初期化（初期化時間も学習時間に含む）。`e_means` は公式デフォルトの初期化、`e_means_pp` は `faiss_kmeans_pp` と同一の k-means++ 初期重心から開始（初期化時間を含む）。以下の列名の emeans_* は、この行の e-means 初期化（candidate）を指す。

| dataset | 基準 | e-means 初期化 | 基準 test MSE | e-means 到達 | 最少到達エポック | その空クラスタ | 同手法の最短到達反復 | e-means最良MSE / 基準 | シード別到達 |
|---|---|---|---|---|---|---|---|---|---|
| imbalanced | faiss_kmeans@25 | e_means_pp | 114.16 | yes | 1 | 9 | 25 | 0.76871 | 5/5 |
| imbalanced | faiss_kmeans_pp@25 | e_means_pp | 83.39 | **NO** |  |  | 25 | 1.0523 | 0/5 |

## 構成別の集計（シード×反復の中央値とばらつき）

| dataset | algorithm | passes | runs | train MSE | test MSE | test min | test max | 空クラスタ 中央値 | 空 max |
|---|---|---|---|---|---|---|---|---|---|
| imbalanced | e_means_pp | 1 | 5 | 91.185 | 93.558 | 92.761 | 96.383 | 9 | 10 |
| imbalanced | e_means_pp | 2 | 5 | 87.532 | 90.422 | 88.837 | 92.309 | 0 | 1 |
| imbalanced | e_means_pp | 3 | 5 | 87.584 | 90.589 | 88.904 | 92.014 | 0 | 0 |
| imbalanced | e_means_pp | 5 | 5 | 87.157 | 89.844 | 88.371 | 91.292 | 0 | 0 |
| imbalanced | e_means_pp | 10 | 5 | 85.282 | 87.755 | 87.338 | 89.372 | 0 | 0 |
| imbalanced | faiss_kmeans | 25 | 5 | 112.44 | 114.16 | 112.99 | 117.92 | 0 | 0 |
| imbalanced | faiss_kmeans_pp | 25 | 5 | 81.495 | 83.39 | 82.113 | 84.446 | 0 | 0 |

## 計時の診断

CPU 時間 / 経過時間 の中央値（プロセス全体・メインスレッド）と、各実行直前の 1 分ロードアベレージ（本プロセス自身のスレッドを含む）。他の回と比べて経過時間だけが伸び、CPU 時間/経過時間 が下がっていればCPU の取り合い（待ち）、比が変わらず時間全体が伸びていればコア自体の遅さ（コア種別・クロック）を示唆する。WSL2 内では CPU クロックは観測できない。「取り合い検出」= メインスレッドの CPU 時間が経過時間の 95% 未満だった実行（学習スレッドは I/O や sleep をしないため、他プロセスに CPU を奪われていたことを示す）。

- ロードアベレージ（1/5/15 分） 開始時: [4.78, 4.28, 2.57]  / 終了時: [5.96, 4.91, 3.15]

| algorithm | runs | プロセスCPU/経過 | メインスレッドCPU/経過 | load 1m 中央値 | load 1m 最大 | 取り合い検出数 |
|---|---|---|---|---|---|---|
| e_means_pp | 25 | 3.753 | 0.981 | 4.69 | 5.85 | 5 |
| faiss_kmeans | 5 | 3.918 | 0.981 | 4.55 | 5.62 | 0 |
| faiss_kmeans_pp | 5 | 3.921 | 0.98 | 5.05 | 5.62 | 2 |

## 注意事項

- MSE は全アルゴリズム共通の厳密 `faiss.knn`（二乗L2）で最終重心に再割当てして算出。e-means の train() が返すラベルは使用していない。
- シード数が少ない場合、中央値・min/max は参考値として扱うこと。
