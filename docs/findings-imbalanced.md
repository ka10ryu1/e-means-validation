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

## 追加調査: d=2 の imbalanced で e-means++ が到達した理由

実施日: 2026-11-12 / 計測コード: `d60528f`（全実行 `repo_dirty: false`）/ 4 スレッド / **品質専用**（`--quality-only`、日中に実行。時間は評価に使っていない）。

規模を振った検証（`reports/scale_20261112-0338_d2_t4/`）では、d=2 の imbalanced で e-means++ が Lloyd++25 に 1 エポックで到達した（最良 / Lloyd++ = 0.976、3/3）。d ≥ 16 では復活が有効な 8 点すべてで未到達だった。ただし d=2 では公式デフォルトによって 2 つの設定が同時に変わる。

- クラスタ復活が無効になる（`unused_threshold=None` → d ≤ 2 では 0）
- バッチの決め方が quality 則になる（`batch_law="auto"` → 低次元では quality、バッチ 512〜4096）

どちらが効いたかを分けるため、d=2・imbalanced（指数 1.3）・n=200,000・k=1,024・3 シード（0, 1, 2）で、2 × 2 の組み合わせを実行した。

| バッチの決め方（バッチサイズ） | 復活の閾値 | Lloyd++25 | e-means++ 1 / 2 / 3 / 5 / 10 エポック | 最良 / Lloyd++ | 到達（1% 以内） | 復活の延べ数（全エポック・全シード） | レポート |
|---|---|---|---|---|---|---|---|
| quality（512〜4096）= d=2 の公式デフォルト | 0 | 0.03945 | 0.03974 / 0.03881 / 0.03883 / 0.03881 / 0.03850 | **0.976** | yes, 1 エポック, 3/3 | 0 | `q_d2imb_quality_thr0` |
| quality（512〜4096） | 0.1 | 0.03945 | 0.05368 / 0.04723 / 0.04757 / 0.04576 / 0.04489 | 1.138 | **NO**, 0/3 | 161〜298 | `q_d2imb_quality_thr01` |
| throughput（4096〜32768） | 0 | 0.03945 | 0.04015 / 0.03963 / 0.03953 / 0.03856 / 0.03889 | **0.978** | yes, 2 エポック, 3/3 | 0 | `q_d2imb_throughput_thr0` |
| throughput（4096〜32768）= d ≥ 64 の公式デフォルトと同じ組み合わせ | 0.1 | 0.03945 | 0.04743 / 0.04617 / 0.04471 / 0.04438 / 0.04324 | 1.096 | **NO**, 0/3 | 90〜119 | `q_d2imb_throughput_thr01` |

test MSE はシードの中央値。バッチサイズはエポック数によって変わる（公式のバッチ則）。復活の延べ数は公式の `n_splits`（学習中に復活させた重心の累計。同じ重心の再復活も数える）で、1 回の学習あたりの値の範囲。どの構成も空クラスタは 0。

- 再現性の確認: 「quality・閾値 0」は、既存の `scale_20261112-0338_d2_t4` の imbalanced の 48 行すべてで、test / train MSE・空クラスタ・バッチ設定がビット単位で一致した（d=2 の公式デフォルトがこの組み合わせであることの確認）。
- 参考: 公式デフォルトの初期化（ランダム）の e-means は、4 通りすべてで Lloyd25 には到達し（3〜5 エポック）、Lloyd++25 には届かなかった（最良 / Lloyd++ = 1.33〜1.40）。こちらは初期化の差が支配的で、復活・バッチの決め方の影響は小さい。

### 結論（この条件の範囲に限る）

6. **d=2 で到達したのは、復活が無効になったためである。バッチの決め方（quality 則）は主因ではない。** 復活を無効にすると、バッチの決め方によらず到達した（0.976 / 0.978）。復活を有効（0.1）にすると、どちらのバッチの決め方でも未到達だった（1.138 / 1.096）。バッチの決め方の差は、到達に要するエポック数（1 対 2）の違いにとどまった。結論 1（主因はクラスタ復活）が、d=64 に加えて d=2 でも、バッチの決め方と切り分けた形で再現した。
7. **k-means++ の初期重心から始めても、復活は多数発動する。** 閾値 0.1 の e-means++ では、1 回の学習で延べ 90〜298 回（k = 1,024 の 9〜29% に相当）の復活が起きた。エポック数を 1 から 10 に増やしても延べ数はほとんど増えず（throughput 則 95〜103 → 110〜114）、quality 則ではむしろ減った（284〜298 → 161〜174）。発動は学習の初期に集中していると考えられる（発動時期は未記録）。
8. quality 則（小さいバッチ）では、復活の延べ数が同じエポック数の throughput 則の約 1.5〜3 倍で、品質の低下も大きかった（1.138 対 1.096）。復活の判定は 5 更新ごとに行われるため、更新回数の多い小さいバッチほど発動の機会が増えると考えられる（未検証）。

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

# 追加調査（d=2）: 既定の設定（初期化・反復・エポック）で実行
for law in quality throughput; do for thr in 0 0.1; do
  uv run python benchmark.py --profile medium --threads 4 --seeds 0 1 2 --dim 2 --datasets imbalanced --quality-only \
      --emeans-batch-law $law --emeans-unused-threshold $thr --run-name q_d2imb_${law}_thr${thr/./}
done; done
```

## 次の検証候補

- 学習中に復活が発動した回数（`n_splits`）と、復活させられた重心の元の質量を記録し、結論 2 のメカニズムを確かめる（`raw_results.csv` の `splits` 列は既に記録あり）。
- 閾値を 0.001〜0.05 で細かく振り、分布ごとの最良値を調べる。
- 復活の発動条件を「平均質量比」ではなく別の基準（例: 一定エポック以上の質量ゼロ）にした場合の比較（公式コードを改変せず、外側から `begin` / `accumulate` / `update` を使う方法を検討）。
- k = 128（Quick）など、クラスタ数を変えた場合の再確認。
