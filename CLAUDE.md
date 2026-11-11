# CLAUDE.md — e-means-validation

Meta FAIR の e-means（`faiss.contrib.e_means`）の主張を、CPU上で独立に検証するリポジトリ。
**目的は e-means の優秀さの証明ではなく、主張がどの条件で再現する／しないかを実測で明らかにすること。**

## 設計思想

- 実測値のみを報告する。未実測の数値、推測値、論文の数値をレポートや README の「結果」に書かない。
- 不利な結果（未到達、倍率 < 1、ばらつき大）も同じ重みで記録する。都合の良い構成だけを選ばない。
- 公式コードは改変しない。`faiss-src/` は `FAISS_COMMIT` の SHA に固定し、Git 管理外。
- 既存検証キット（`docs/validation-kit.md`）のロジックを尊重する。データ生成・スコアリング・品質一致規則を変更する場合は、理由と影響を docs に記録し、過去結果と比較不能になることを明記する。
- 最小限の依存（numpy, faiss-cpu, pytest）。フレームワーク・外部サービス・テレメトリを入れない。

## 構成

| パス | 役割 |
|---|---|
| `benchmark.py` | CLI。スレッド環境変数を numpy/faiss import 前に設定する（この順序を崩さない） |
| `make_report.py` | `results/<run>/raw_results.csv` から集計・レポートを再生成 |
| `emv/official.py` | 公式ソースの読み込みと読み込み元・SHA・ハッシュ検証 |
| `emv/data.py` | データ生成（キットと同一）、プロファイル |
| `emv/runners.py` | 計時付き学習、厳密スコアリング |
| `emv/analysis.py` | 集計、品質一致倍率、Markdown レポート |
| `scripts/setup_faiss.sh` | 固定SHAの公式ソース取得 |
| `results/` | 生データ（Git 管理外） |
| `reports/<run>/` | 集計CSV・metadata.json・report.md（Git 管理対象） |

## 実装ルール

- 計時は `train()` 呼び出しのみ。データ生成・評価・初期重心作成を計時に含めない。
- 品質評価は全アルゴリズム共通の厳密 `faiss.knn` で最終重心に対して行う。アルゴリズムが返すラベルや内部目的関数を品質指標に使わない。
- 比較は同一データ・同一 k・同一次元・同一スレッド数で行う。新アルゴリズムを追加する場合も `emv/runners.py` に `run_xxx(…) -> (centers, seconds, info)` の形で追加し、同じ評価経路を通す。
- 例外を握りつぶさない。不正な重心（NaN、形状不一致）はエラーにする。
- 公式APIの変更に追従する場合は、変更点と理由を `docs/validation-kit.md` の「公式APIとの差分確認」に追記し、`FAISS_COMMIT` を更新、`tests/test_official.py` を通す。
- `FAISS_COMMIT` を更新したら、以前のレポートとは別条件であることを明記する。
- 新しい指標・判定規則を加えたら `docs/methodology.md` と README の評価指標節を更新する。
- 商用・業務データは使わない（`--input` は公開データのみ）。

## よく使うコマンド

```bash
uv sync
bash scripts/setup_faiss.sh
uv run pytest
uv run python benchmark.py --profile quick --threads 4
uv run python benchmark.py --profile medium --threads 4
uv run python make_report.py results/<run-name>
```

## 性能計測の注意

- **速度（時間・倍率）の計測は、ユーザーの作業がない時間帯に `scripts/run_speed_suite.sh` で行う。** 作業中の実行は IDE などとの CPU の取り合いで時間がずれる（実測で確認済み）。
- 作業時間中に実験する場合は `--quality-only` を付け、品質（MSE・空クラスタ・最少到達エポック）だけで結論を出す。時間・倍率を報告しない。
- レポート冒頭に「CPU の取り合いを検出」の警告がある結果は、速度の根拠に使わない。
- GitHub Actions など共有ランナーで性能を評価しない（CI は正しさのテストのみ）。
- 計測中は他の重い処理を同時に走らせない。複数のベンチマークは順番に実行する。
- レポートを Git に追加する前に `metadata.json` の `repo_dirty` が false であることを確認する（コードをコミットしてから計測する）。

## 夜間の速度計測の後にやること

1. `bash scripts/run_speed_suite.sh` の出力で、各実行の `contention:` が `none detected` であることを確認する。検出があれば、その実行は速度の根拠に使わず再計測する。
2. `reports/speed_<日時>_quick_t4/` と `reports/speed_<日時>_medium_t4/` の `report.md` を読み、`metadata.json` の `repo_dirty: false` と `faiss_source_matches_pin: true` を確認する。
3. README の「7. 実測結果」の速度の数値（秒・倍率）を、夜間計測の値に差し替える。日中計測の注記（取り合い・計時の再現性）は、夜間の結果と比べたうえで残すか書き直す。品質（MSE・到達・空クラスタ）は計測時刻に依存しないので、値が一致することを確認するだけでよい。
4. 解釈（倍率、1 エポックあたりのコスト比など）の数値も夜間の値で見直し、変わった結論は明記する。
5. レポートと README をコミットし、PR のブランチに push する。
