# GPU 解析の上限測定（日本語）

最終更新: run 53 相当。対象: x265 2.6 / x264 0.152.2854。
測定スクリプト: `tools/ceiling_x265.py`。

## 1. 方法

x265 の analysis save/load を使い、**解析結果を外部から与えた場合の速度上限**を測る。

- save: 通常の解析を行い `--analysis-reuse-file` に保存。
- load0: `--analysis-reuse-mode load --refine-inter 0 --refine-intra 0 --no-refine-mv`
- load1: `refine-inter 1 / refine-intra 1`
- load3: `refine-inter 3 / refine-intra 3 / --refine-mv`
- 各点 **3 回**の中央値。出力サイズと MD5 を記録。
- load のビットストリームは save と **MD5 一致**（BD-rate は 0%）。

実行例:

```sh
python3 tools/ceiling_x265.py --y4m build/work/cockL.y4m --label cockL \
  --frames 240 --presets medium,slow,slower --crfs 23,27,31 --reps 3
python3 tools/ceiling_x265.py --y4m build/work/1080p.y4m --label 1080p \
  --frames 12 --presets medium,slow,slower --crfs 27 --reps 3
```

## 2. 640x352 / cockL 240f（中央値、秒）

| preset | CRF | save | load0 | load1 | load3 | save/load0 | 出力 |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 23 | 2.008 | 1.685 | 1.683 | 1.685 | 1.19x | 同一 |
| medium | 27 | 1.816 | 1.524 | 1.580 | 1.503 | 1.19x | 同一 |
| medium | 31 | 1.705 | 1.426 | 1.417 | 1.416 | 1.20x | 同一 |
| slow | 23 | 3.547 | 2.972 | 3.025 | 3.076 | 1.19x | 同一 |
| slow | 27 | 3.139 | 2.679 | 2.674 | 2.665 | 1.17x | 同一 |
| slow | 31 | 2.943 | 2.454 | 2.446 | 2.495 | 1.20x | 同一 |
| slower | 23 | 12.792 | 11.947 | 11.832 | 11.668 | 1.07x | 同一 |
| slower | 27 | 10.591 | 9.713 | 9.995 | 12.119 | 1.09x | 同一 |
| slower | 31 | 10.524 | 9.016 | 8.551 | 8.884 | 1.17x | 同一 |

- load1/load3 は load0 とほぼ同じ。解析再利用だけでは追加の画質改善は出ない
  （今回の設定では同一ストリーム）。
- CRF による差は小さい。preset が遅いほど解析の相対コストが下がり、
  reuse の速度利得は **1.07–1.20x** に縮む。

## 3. 1920x1080 / 12f（中央値、秒）

| preset | CRF | save | load0 | load1 | load3 | save/load0 |
|---|---:|---:|---:|---:|---:|---:|
| medium | 23 | 0.807 | 0.619 | 0.628 | 0.626 | 1.30x |
| medium | 27 | 0.797 | 0.585 | 0.589 | 0.606 | 1.36x |
| medium | 31 | 0.807 | 0.566 | 0.573 | 0.556 | 1.43x |
| slow | 27 | 0.971 | 0.763 | 0.732 | 0.718 | 1.27x |
| slower | 27 | 2.060 | 1.909 | 1.851 | 1.785 | 1.08x |

1080p は 12 フレームの短尺のため、固定オーバーヘッドの影響が大きい。
それでも **medium 1.30–1.43x / slow 1.27x / slower 1.08x** で、
解析を完全に無料にしても 2x には届かない。

## 4. 上限の解釈

### x265
- 実測の解析再利用上限は **1.1–1.4x**。
- 目標 **1.5–2x** は、analysis reuse だけでは届かない。
  - 追加の ME 網羅探索、CU 分割枝刈り、前処理、MCSTF などを積む必要がある。
  - 1080p では medium 1.36x がまず上限。1.5x を超えるには解析以外の短縮が必要。

### x264（推定）
- medium/slow の先読み+ME は全体の 3–5 割というレビュー前提。
- 理論上限は 1.4–2x。ただし x265 の実測が 1.1–1.4x だったことから、
  実装オーバーヘッドを考えると **1.2–1.5x 目標は上限内だが余裕は小さい**。
- slow/slower では解析の相対コストが下がるため、1.5x は難しい。

### 判定
| 目標 | 上限内か | コメント |
|---|---|---|
| x264 medium 1.2–1.5x | **条件付き YES** | 1.2–1.3x は現実的。1.5x は解析無料でも厳しい |
| x264 slow 1.2–1.5x | **条件付き YES（下端）** | 1.2x 前後が上限 |
| x265 medium 1.5–2x | **NO（reuse のみ）** | 実測 1.3–1.4x。追加の枝刈りが必要 |
| x265 slow 1.5–2x | **NO（reuse のみ）** | 実測 1.27x |

## 5. x264 ESA MV 注入の設計メモ（実装しない）

対象: x264 0.152.2854（`/usr/bin/x264`）。ソースは未取得のため、
以下は公開 API/既知の `me.c` 構造に基づく設計。

### 注入点
- `encoder/me.c` の `x264_me_search_ref(x264_t *h, x264_me_t *m,
  int16_t (*mvc)[2], int i_mvc, int *p_halfpel_thresh)`。
- ESA の候補評価は `COST_MV(x, y)` マクロで行われる。外部候補を
  **探索の開始点**として渡すなら、`m->mv[0] = hint_x; m->mv[1] = hint_y;`
  を設定してから ESA を呼ぶ。
- 追加候補として評価するなら、ESA の後に
  `COST_MV(hint_x, hint_y);` を 1 回呼び、最良なら `m->mv` が更新される。
  `COST_MV` は `m->cost`（SATD + MV コスト）を更新するので、
  CPU の最終 RD 判定は従来どおり。
- 依存を切るため、ヒントは **前フレームの同位置 MB の MV** または
  **低解像度解析の MV** を整数倍したものを使う。隣接 MV 予測には依存しない。

### esa MV のダンプ
- `--me esa` の結果は `m->mv[0..1]` と `m->cost`。
- `x264_me_search_ref()` の return 直前に、環境変数
  `X264_MV_DUMP=<file>` があれば
  `frame_num, mb_x, mb_y, m->mv[0], m->mv[1], m->cost` を追記する。
- フレーム境界は `x264_encoder_encode()` の呼び出し側で区切る。
- 出力はテキスト 1 行/MB。読み戻しは同じ形式。

### 最小パッチ案（数十行）
1. `common/base.c` に `static FILE *mv_dump, *mv_hints;`
2. `x264_me_search_ref()` の入口で、hint があれば
   `m->mv[0] = hint[0]; m->mv[1] = hint[1];` を設定。
3. ESA の後に `COST_MV(hint[0], hint[1]);` を追加（範囲内のみ）。
4. return 前に dump を 1 行書く。
5. `x264_param_t` ではなく環境変数で ON/OFF（ABI を変えない）。
6. ヒントの無い MB / フレームは完全に従来経路。

### 実装可否
- **可能**。ただし x264 の GPL フォークと、`COST_MV` のスコープ・
  `m->i_pixel`・`m->p_fref` の前提を確認する必要がある。
- 本タスクでは路線未確定のためパッチは実装しない。

## 6. まとめ

- x265 の解析再利用上限は **1.1–1.4x**（1080p medium 1.36x、slower 1.08x）。
- 「GPU が無限に速い」上限を解析だけに求めると、x265 1.5–2x は届かない。
- x264 1.2–1.5x は上限内だが、1.5x 側は余裕が小さい。
- 次段は「解析の無料化」だけでなく、**候補生成・枝刈り・前処理**を
  組み合わせたパレート改善が必要。
