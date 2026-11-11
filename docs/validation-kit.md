# 既存検証キット（e_means_validation_kit.zip）からの移行記録

## 元のキットの構成

| ファイル | 内容 |
|---|---|
| `README.md` | venv + pip での実行手順、評価指標・公平性の説明 |
| `requirements.txt` | `faiss-cpu>=1.9`, `numpy>=1.26` |
| `benchmark.py` | 単一ファイルのベンチマーク（264行） |
| `__pycache__/benchmark.cpython-313.pyc` | ビルド生成物（移行対象外） |

## 維持したロジック（変更なし）

| キットの関数 | 移行先 | 備考 |
|---|---|---|
| `synthetic_data` | `emv/data.py` | 生成式・乱数シード `20261111` とも同一。同じ設定なら同じデータになる |
| `real_embeddings` | `emv/data.py` | 同一 |
| `load_official_emeans` | `emv/official.py` | `faiss.contrib.__path__` 先頭に公式ソースを差し込む方式を踏襲。読み込み元検証を強化（後述） |
| `score` / `validate_centers` | `emv/runners.py` | 同一（厳密 `faiss.knn` による最終重心への再割当て） |
| `run_faiss_kmeans` | `emv/runners.py` | `max_points_per_centroid=n`（サンプリング上限解除）を維持。`init_centroids` 引数を追加 |
| `run_emeans` | `emv/runners.py` | 同一。`alpha0`, `batch_law` の記録と `init_centroids` 引数を追加 |
| `summarize` の品質一致判定 | `emv/analysis.py: compare` | 「中央値 test MSE ≤ Lloyd25 中央値 × 1.01 のうち最短時間」を維持 |
| CSV / metadata.json 出力 | `benchmark.py` | 列は上位互換（キットの列はすべて残している） |
| `PROFILES` quick / medium | `emv/data.py` | 同一。テスト用に `smoke` を追加 |

## 追加・変更した点と理由

| 変更 | 理由 |
|---|---|
| venv+pip → uv (`pyproject.toml`, `uv.lock`, `.python-version`=3.11) | 要件。`requirements.txt` は移行していない（不要） |
| 公式ソースを `FAISS_COMMIT` のSHAに固定し `scripts/setup_faiss.sh` で取得 | キットは `git clone --depth 1`（HEAD）で、再現性がなかった |
| 固定SHAと不一致、または `contrib/e_means` に変更がある場合は実行停止（`--allow-unpinned` で解除） | 意図しないコード差での計測を防ぐ |
| 読み込み元検証を `NumpyEmeans` のモジュールと `base` まで拡張。事前にwheel版がimport済みなら拒否 | faiss-cpu 1.16.0 のwheelに `faiss.contrib.e_means` が同梱されているため、取り違えの可能性がある |
| 公式ソースとwheel同梱版のハッシュを metadata に記録 | どちらを使っても同じコードかを後から確認できるようにする |
| スレッド数を `OMP/OPENBLAS/MKL_NUM_THREADS` でも固定（numpy/faiss の import 前） | キットは `faiss.omp_set_num_threads` のみで、e-means 内の numpy 演算のスレッドが未固定だった |
| ウォームアップ（各データセットで両アルゴリズムを1エポック、計測外） | 初回実行時のスレッドプール起動などの影響除去 |
| 各 (dataset, seed) ブロック内で構成の実行順をシャッフル（記録あり） | キットは常に k-means → e-means の固定順だった |
| `--repeats` | 同一seedでの計時ノイズの把握 |
| k-means 反復数 `5 10 25`、e-means エポック数 `1 2 3 5 10` に拡張（キットは k-means 25 / e-means 2 3 5） | k-means 自身が25反復より前に収束している場合に倍率を過大評価しないため。e-means がどのエポックで到達するかを細かく見るため |
| `--init shared` | 初期値の違いによる交絡を取り除く比較を可能にする |
| k-means++ 初期重心から始める e-means（`e_means_pp`, `--emeans-inits`）を追加 | 初期化の効果と更新則の効果を切り分けるため。初期重心は `faiss_kmeans_pp` とビット単位で一致 |
| k-means++ 初期化の Faiss k-means（`faiss_kmeans_pp`）を2つ目の基準として追加（`--kmeans-inits`） | ランダム初期化の Lloyd が局所解に陥ると e-means の品質面の優位が過大に見えるため。キットの Lloyd25（ランダム初期化）基準は維持 |
| シード別の到達判定・倍率（`per_seed_*`） | 中央値どうしの比較だけでは隠れるばらつきを示す |
| `comparison.csv`, `report.md`, `reports/<run>/` | 集計をGit管理できる形で保存 |
| e-means のエポック数を float でも受け付け | 公式は小数エポックをサポート |

## 公式APIとの差分確認（FAISS_COMMIT 時点）

- `NumpyEmeans(d, k, nepochs=5, batch_size=None, alpha0=None, batch_law="auto", seed=1234, verbose=False, ..., max_points_per_centroid=None)` — キットが使う引数（`seed`, `nepochs`, `verbose`）はそのまま有効。
- `train(x, init_centroids=None)` → `(centroids, labels)`、`recipe["batch_size"]`、`n_splits` — キットの想定どおり。
- 2026-11-06 に `niter` → `nepochs` の改名（faiss #5848）があった。キットはすでに `nepochs` を使っており、変更不要。
- 2026-11-09 に `max_points_per_centroid` が追加（#5855）。既定 `None`＝全件学習なので、本検証では指定しない（k-means 側もサンプリング上限を解除済み）。
- 公式 `benchs/e_means/README.md` には「e-means はまだ faiss のリリースに含まれていない」とあるが、faiss-cpu 1.16.0 のwheelには同梱されており、`*.py` は固定SHAのソースと一致した（metadata の `bundled_wheel_e_means_identical`）。
