## M-9D リトライ結果（未完了）

実装:
- `x264_t` に `aeg_in_encode` / `aeg_enc_frame_idx` を追加。
- `encoder/encoder.c` の `slice_write()` の MB ループ前後で
  `aeg_mv_hint_begin_frame()` / `end_frame()`。
- `X264_MV_HINT_MODE=extra|seed|skip|limit:<N>`、キーは自前の
  単調増加フレーム番号 + mbx/mby/ref/i_pixel。
- `X264_MV_HINT_STATS=1` で hits/misses/applied を atexit 出力。
- dump の `fprintf` を mutex 保護、quarter-pel→fullpel、±16 クランプ。

現状:
- env 未設定は本家と MD5 一致（1080p 12f）。
- dump が **出力されない**（`begin_frame` が実際の ME 呼び出しに
  到達していない、または `g_in_encode` がワーカーで見えていない）。
- `X264_MV_DUMP` のファイルが生成されず、seed/skip/limit の
  安全な実測に至っていない。
- よって時間短縮率・BD-rate は **未測定**。

原因候補:
- x264 の frame/slice threads で `slice_write()` を実行する `x264_t` と
  `x264_me_search_ref()` を実行する `x264_t` が別インスタンス。
- `aeg_in_encode` を `h` に持たせても、ME 側のワーカー h には伝播しない。
- 次はプロセス全体のフラグではなく、**ME を実行する h を特定して
  begin/end をそこで呼ぶ**必要がある。
