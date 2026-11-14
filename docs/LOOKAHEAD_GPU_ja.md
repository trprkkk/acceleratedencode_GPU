# 先読みコストの GPU 化（M-9K 設計メモ）

## 1. 実測（640p cockL / 240f / threads 4 / 3回中央値）

| preset | 構成 | 中央値 | 出力 |
|---|---|---:|---:|
| medium | base | **11.24 s** | 99,967 |
| medium | `--no-mbtree` | 14.72 s | 200,853 |
| medium | `--rc-lookahead 0` | 14.62 s | 200,853 |
| medium | `--rc-lookahead 10` | 11.19 s | 94,741 |
| medium | `--no-scenecut` | 11.02 s | 99,966 |
| slow | base | **14.07 s** | 100,905 |
| slow | `--no-mbtree` | 16.76 s | 182,757 |
| slow | `--rc-lookahead 0` | 16.36 s | 182,757 |
| slow | `--rc-lookahead 10` | 13.20 s | 91,822 |
| slow | `--no-scenecut` | 13.03 s | 100,904 |

- `--no-mbtree` / `--rc-lookahead 0` は **ビットレートが倍増**し、
  エンコード自体が遅くなる。先読みコストの単純な差分には使えない。
- `--rc-lookahead 10` は medium でほぼ同じ、slow で **-0.87 s（-6%）**。
- `--no-scenecut` は medium -0.22 s、slow -1.04 s。
- **先読み（slicetype + mbtree）の時間寄与は medium で数 %、
  slow で 6–7% 程度**と推定。
- 先読みが 0 秒になっても **medium 1.00–1.02x / slow 1.06–1.07x** が上限。

## 2. 入出力仕様

対象:
- `x264_slicetype_frame_cost()` / `x264_slicetype_mb_cost()`
- `x264_mbtree_propagate_cost()`（固定小数点）

入力:
- lowres プレーン（1/4 解像度、Y のみ、パディング付き）
- 参照フレーム lowres リング
- QP / lambda / 重み
- MB ごとの intra/inter コスト、SATD

出力:
- `i_cost_est[SLICE_TYPE_COUNT]`
- `i_cost_est_b[]`（B 参照）
- mbtree の `propagate_amount`（int32）

## 3. ビット一致の見込み

- slicetype の SATD は整数 Hadamard。GPU `satd_batch` と同じ順序・
  丸めにすれば **ビット一致可能**。
- mbtree は固定小数点（int32/int64）で、加算順序に依存。
  MB 順・参照順を固定すれば一致可能。
- ただし x264 は lowres を **8x8 SATD + 独自コスト**で評価し、
  `i_cost_est` に lambda を掛ける。GPU 側で同じ整数演算を再現する
  必要があり、**1 bit でも違うとフレームタイプが変わり BD-rate が跳ねる**。
- 見込み: **可（実装コスト大）**。整数演算のみで、浮動小数は無い。

## 4. 実装案

- 1 フレーム 1 起動 + 1 同期。
- lowres プレーンはデバイス常駐（リング 96 スロット）。
- 出力は MB ごとの `cost`（int32）と `propagate`（int32）。
  - 1080p lowres（480x270）: 8,160 MB × 8 B = **65 KB/frame**。
  - 640p: 920 MB × 8 B = **7.4 KB/frame**。
- 転送は ≤128 KB/frame を満たす。

### カーネル一覧と工数

| カーネル | 内容 | 工数 |
|---|---|---:|
| `lowres_satd_batch` | 8x8/16x16 SATD | 2–3 日 |
| `slicetype_cost` | intra/inter/B コスト | 4–5 日 |
| `mbtree_propagate` | 固定小数点伝播 | 3–4 日 |
| リング同期 | lowres upload / 1 起動 | 1–2 日 |
| ビット一致テスト | 240f × 4 クリップ | 2–3 日 |
| **合計** | | **12–17 日** |

## 5. 危険度

- ビット一致が取れない場合、フレームタイプ・QP・mbtree が変わり、
  BD-rate は **+2〜+10%** 悪化し得る（先読みは全体に波及）。
- 部分一致（例: slicetype のみ）でも、mbtree が変わると
  ビット配分が変わる。
- 先読みの時間寄与は medium 数 %、slow 6–7%。
  **GPU 化の理論上限は medium 1.02x / slow 1.07x**。

## 6. 推奨

- **先読み GPU 化は投資対効果が低い**（上限 1.02–1.07x、工数 12–17 日、
  ビット一致リスク大）。
- **x265 アダプタへ切り替える**方を推奨。
  x265 は `--analysis-load` / `--refine-inter` があり、
  GPU ヒントの受け口が既にある。slow 相当のパレート改善を
  低リスクで狙える。
- 先読み GPU 化は、medium の 2% をどうしても削りたい場合の
  最終手段として保留。
