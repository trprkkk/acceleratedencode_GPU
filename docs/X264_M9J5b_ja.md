## M-9J5b dump を encoder.c の MB ループへ移設

### 実装
- `aeg_mv_hint_dump_mb(h, mbx, mby)` を追加。
  `x264_macroblock_analyse()` 直後に呼ぶ設計。
  - 座標はループ変数 `i_mb_x/i_mb_y`。
  - MV は `h->mb.mv[0][mb_xy][0/1]`。
  - cost は 0、ref は 0、i_pixel は 3。
  - mutex で保護。
- `aeg_mv_hint_post()` から dump を削除。

### 結果
- 呼び出しを有効にすると **SIGSEGV**。
  `h->mb.mv[0]` の参照、または `h->mb.ref[0]` の参照で落ちる。
- `h->mb.mv[0]` の null ガード、`ref` の固定 0 化でも再現。
- 安全のため呼び出しは無効化。既定ビルドは本家と MD5 一致。

### 現状
| 項目 | 結果 |
|---|---|
| dump 移設 | 実装したがクラッシュ |
| 有効キー 211,200 | **未達** |
| 一致率 | **未測定** |
| 既定 MD5 | PASS |

### 原因候補
- `h->mb.mv[0]` は MB 配列ではなく、別のポインタ型として解釈されている。
- `mb_xy` の計算（`mby * i_mb_stride + mbx`）が analyse 直後の値と不一致。
- `h->mb.mv` の有効期間が `x264_macroblock_encode()` 前で終わる。

### 次
- `h->mb.mv[0]` の実体を確認し、`x264_mb_cache_mv()` 相当の
  マクロを使う。
- または `x264_macroblock_analyse()` 内の `h->mb.cache` から
  MB 単位 MV を取る。
- dump 修正後に一致率を再測定。
