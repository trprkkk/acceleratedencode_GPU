## M-9C ヒント開始点/打ち切りモード（未完了）

実装したもの:
- `X264_MV_HINT_MODE=extra|seed|skip|limit:<N>`
- キー付き lookup の試作（frame, mbx, mby, ref, i_pixel）
- dump を 8 列化、mutex でスレッド間の行混在を防止
- `skip` はヒント評価後に fullpel 探索を飛ばす、`limit:N` は探索半径を N に置換

現状:
- **env 未設定では本家と MD5 一致**（1080p 12f cockL）。
- ヒント有効時は x264 の内部 frame id が lookahead と本エンコードで
  安定せず、dump に負/巨大な frame 値が混入する。
- `h->fenc->i_frame < 0` で除外しても、キーが一致せず
  `seed/skip/limit` はクラッシュまたは出力破損。
- したがって **時間短縮率と BD-rate の実測は未完了**。

原因:
- `x264_me_search_ref()` は lookahead の解析でも呼ばれ、その `x264_t` は
  本エンコードの frame/mb 状態を持たない。
- frame id として `i_frame`, `i_frame_num`, `h->fenc->i_frame` のいずれも
  lookahead 中は安定しない。
- 正しいキーは「本エンコードのフレーム・MB 順」を呼び出し側で付与するか、
  lookahead 呼び出しを明示的に除外する必要がある。

次段:
1. `h->fenc->i_type == X264_TYPE_AUTO` などで lookahead を除外。
2. 本エンコード時だけ dump/hint を有効化。
3. キー付き lookup の一致率をログ化（hit/miss）。
4. `seed` の探索開始点変更が x264 の `bmx/bmy` に正しく反映されるか確認。
5. その後、時間短縮と BD-rate を測定。
