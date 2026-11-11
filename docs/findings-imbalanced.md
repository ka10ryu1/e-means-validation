# 調査: 不均衡データで e-means が Lloyd++ に劣後する原因

実施日: 2026-11-11 / 計測コード: `a3a6a19`（全実行 `repo_dirty: false`）/ 4 スレッド
すべて **品質専用の計測**（`--quality-only`）。日中に他の作業と並行して実行したため、時間は評価に使っていない。
MSE は実行環境の負荷に依存しない（同一コード・同一スレッド数で再実行するとビット単位で一致することを確認済み）。

## 背景

Medium（200,000 × 64 次元、k = 1,024）の imbalanced データでは、Lloyd++（k-means++ 初期化の Faiss k-means）と **同一の初期重心** から始めた e-means（e-means++）でも、10 エポックで test MSE 103.4 と、Lloyd++25 の 83.4 に届かなかった（1.24 倍悪い、5 シードすべて未到達。`reports/medium_t4/`）。初期化をそろえているため、差は更新則による。

## 仮説

e-means は 5 更新ごとに、質量が **平均質量の 0.1 倍未満** の重心を「使われていない」とみなし、質量の大きい重心を分割して復活させる（公式 `unused_threshold`、既定 0.1。`faiss-src/contrib/e_means/base.py`）。
imbalanced データ（クラスタサイズ ∝ rank^-1.3）では、k = 1,024 のとき **真のクラスタの 55%（データの 3.2%）が平均質量の 0.1 倍未満** になる。小さな真のクラスタに置かれた重心が「使われていない」と判定され、引き剥がされ続けるのではないか。

予測: (1) 復活を無効化（`unused_threshold=0`）すると差が縮む。(2) 小さなクラスタがほぼない分布では差が出ない。

## 結果（Medium、5 シード、e-means++ 対 Lloyd++25、test MSE の中央値）

| 条件 | 指数 | 閾値 0.1 未満の真のクラスタ | Lloyd++25 | e-means++ 1 / 2 / 3 / 5 / 10 エポック | 最良 / Lloyd++ | 到達（1% 以内） | レポート |
|---|---|---|---|---|---|---|---|
| 公式デフォルト | 0.5 | 0% | 116.8 | 112.7 / 108.9 / 107.3 / 106.6 / 105.1 | **0.900** | yes, 1 エポック, 5/5 | `q_imb_exp05` |
| 公式デフォルト | 1.0 | 0% | 103.0 | 124.1 / 116.4 / 118.6 / 115.4 / 113.6 | 1.102 | **NO**, 0/5 | `q_imb_exp10` |
| 復活なし（閾値 0） | 1.0 | | 103.0 | 108.7 / 104.3 / 104.2 / 102.5 / 103.1 | **0.995** | yes, 5 エポック, 5/5 | `q_imb_exp10_thr0` |
| 公式デフォルト | 1.3 | 55% | 83.4 | 107.9 / 104.7 / 105.1 / 104.1 / 103.4 | 1.239 | **NO**, 0/5 | `medium_t4` |
| 閾値 0.01 | 1.3 | | 83.4 | 91.2 / 87.8 / 88.6 / 87.8 / 87.5 | 1.049 | **NO**, 0/5 | `q_imb_thr001` |
| 復活なし（閾値 0） | 1.3 | | 83.4 | 88.9 / 86.6 / 86.5 / 85.1 / 84.7 | **1.016** | **NO**（僅差）, 1/5 | `q_imb_thr0` |
| `batch_law="quality"` | 1.3 | | 83.4 | 127.7 / 117.3 / 118.8 / 113.4 / 111.6 | 1.338 | **NO**, 0/5 | `q_imb_batchq` |
| 公式デフォルト | 1.6 | 81% | 69.6 | 83.5 / 81.9 / 82.8 / 82.0 / 81.3 | 1.169 | **NO**, 0/5 | `q_imb_exp16` |
| 復活なし（閾値 0） | 1.6 | | 69.6 | 72.9 / 72.0 / 72.0 / 71.0 / 71.2 | **1.020** | **NO**（僅差）, 1/5 | `q_imb_exp16_thr0` |

均等サイズ（separated、指数 0 に相当）での副作用の確認:

| 条件 | Lloyd++25 | e-means++ 1 / 2 / 3 / 5 / 10 エポック | 最良 / Lloyd++ | 到達 | レポート |
|---|---|---|---|---|---|
| 公式デフォルト | 121.3 | 115.5 / 113.4 / 111.7 / 109.0 / 105.0 | **0.866** | yes, 1 エポック, 5/5 | `medium_t4` |
| 復活なし（閾値 0） | 121.3 | 126.0 / 126.0 / 126.0 / 123.5 / 122.3 | 1.009 | yes, 10 エポック, 4/5 | `q_sep_thr0` |

参考: 公式デフォルトの初期化（ランダム）の e-means で復活を無効化すると、指数 1.3 で最良 117.1 → 120.3 と **悪化** した（`q_imb_thr0`）。

## 結論（この条件の範囲に限る）

1. **不均衡データでの劣後の主因は、クラスタ復活（`unused_threshold`）である。** 復活を無効化すると、指数 1.0 では Lloyd++ に到達し（0.995）、1.3 / 1.6 でも差は 1.6〜2.0% まで縮んだ（1.24 / 1.17 倍から）。
2. **ただし真のクラスタの大小だけでは説明しきれない。** 指数 1.0 では閾値を下回る真のクラスタが 0% なのに 10% 劣後し、復活の無効化で解消した。学習中に重心の質量が閾値を下回る（例: 1 つの真のクラスタを 2 つの重心が分け合う）ことでも復活が発動していると考えられる（未検証）。
3. **復活は分布によって効果の符号が逆になる。** 均等サイズの separated では、復活ありの方がよく（0.866 対 1.009）、ランダム初期化の e-means でも復活が初期化の弱さを補っていた。不均衡データでは逆に大きく損をする。
4. **公式の「追加のハイパーパラメータ不要」という主張に対する反例。** `unused_threshold` は既定値のままだと不均衡データで 10〜24% の品質低下を招き、どの値が最良かはデータ分布に依存する。`batch_law="quality"` はこの問題を改善しなかった（悪化した）。
5. 復活を無効化しても、指数 1.3 / 1.6 では「1% 以内」の基準にはわずかに届かない（シード別 1/5）。残りの差の原因は未特定。

## 再現コマンド

```bash
C="--profile medium --threads 4 --quality-only --kmeans-inits random kmeanspp --kmeans-niters 25 --emeans-epochs 1 2 3 5 10"
uv run python benchmark.py $C --datasets imbalanced --emeans-inits random kmeanspp --emeans-unused-threshold 0 --run-name q_imb_thr0
uv run python benchmark.py $C --datasets imbalanced --emeans-inits kmeanspp --emeans-unused-threshold 0.01 --run-name q_imb_thr001
uv run python benchmark.py $C --datasets imbalanced --emeans-inits kmeanspp --emeans-batch-law quality --run-name q_imb_batchq
uv run python benchmark.py $C --datasets imbalanced --emeans-inits kmeanspp --imbalance-exponent 0.5 --run-name q_imb_exp05
uv run python benchmark.py $C --datasets imbalanced --emeans-inits kmeanspp --imbalance-exponent 1.0 --run-name q_imb_exp10
uv run python benchmark.py $C --datasets imbalanced --emeans-inits kmeanspp --imbalance-exponent 1.0 --emeans-unused-threshold 0 --run-name q_imb_exp10_thr0
uv run python benchmark.py $C --datasets imbalanced --emeans-inits kmeanspp --imbalance-exponent 1.6 --run-name q_imb_exp16
uv run python benchmark.py $C --datasets imbalanced --emeans-inits kmeanspp --imbalance-exponent 1.6 --emeans-unused-threshold 0 --run-name q_imb_exp16_thr0
uv run python benchmark.py $C --datasets separated --emeans-inits kmeanspp --emeans-unused-threshold 0 --run-name q_sep_thr0
```

## 次の検証候補

- 学習中に復活が発動した回数（`n_splits`）と、復活させられた重心の元の質量を記録し、結論 2 のメカニズムを確かめる（`raw_results.csv` の `splits` 列は既に記録あり）。
- 閾値を 0.001〜0.05 で細かく振り、分布ごとの最良値を調べる。
- 復活の発動条件を「平均質量比」ではなく別の基準（例: 一定エポック以上の質量ゼロ）にした場合の比較（公式コードを改変せず、外側から `begin` / `accumulate` / `update` を使う方法を検討）。
- k = 128（Quick）など、クラスタ数を変えた場合の再確認。
