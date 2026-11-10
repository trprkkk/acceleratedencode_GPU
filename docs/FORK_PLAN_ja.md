# フォーク計画（x264/x265 + GPU 解析アダプタ）

日付: run 58 相当。方針: 案C（1080p+ / medium〜slow のパレート改善）。

## 0. 現在の状態

**x264 ソースの取得がサンドボックスに拒否されたため、フォークは未初期化**。
`forks/x264/` には README と適用スクリプトの設計のみを置いている。
ネットワーク許可またはソース提供後、以下を実行する。

```sh
cd forks
git clone --depth 1 --branch stable https://code.videolan.org/videolan/x264.git x264
cd x264 && git rev-parse HEAD > ../X264_REVISION
./configure --enable-static --enable-pic && make -j"$(nproc)"
python3 ../../tools/apply_x264_hint.py --src .
make -j"$(nproc)"
```

## 1. リポジトリ構成

```
forks/
  README_ja.md
  X264_REVISION          # 取得後に記録
  x264/                  # 上流 + 最小パッチ
    encoder/me.c         # x264_me_search_ref() に 2 箇所だけ挿入
    encoder/aeg_mv_hint.{c,h}
  x265/                  # 次段（未着手）
tools/
  apply_x264_hint.py     # 冪等なパッチ適用（--revert 可）
  go_gate.py             # GO ゲート測定
  ceiling_all.py         # 上限測定
docs/
  FORK_PLAN_ja.md
  GO_GATE_SUMMARY_ja.md
```

## 2. アダプタ IF（候補 MV 配列）

GPU 解析エンジンと x264/x265 の間で受け渡す形式。

### 入力ヒント

- 1 ヒント = `int16_t mv_x, mv_y`（整数ルーマサンプル単位）。
- 範囲: `-32768..32767`。実用上は `±search_range`。
- フレームあたり `mb_w * mb_h` 個、ラスタ順。
- 追加候補を許す場合: `cost_q8`（SATD の 1/256 単位）と
  `partition`（16x16=0, 16x8=1, 8x16=2, 8x8=3）を付ける。
- 候補数はプリセットが使う上位 1〜2 個に制限（転送量削減）。

### 出力（dump）

`X264_MV_DUMP` の 1 行:
```
mv_x mv_y cost i_pixel
```
- `mv_x/mv_y`: 整数ルーマ。
- `cost`: x264 の SATD + MV コスト（内部単位）。
- `i_pixel`: x264 の partition 種別。
- フレーム境界は呼び出し側で区切る。

## 3. パッチ方針

- `x264_me_search_ref()` の入口で `aeg_mv_hint_pre(m)`。
  - `X264_MV_HINT` があるときだけ `m->mv[0..1]` を seed に設定。
- 関数出口で `aeg_mv_hint_post(m)`。
  - `X264_MV_DUMP` があるときだけ 1 行追記。
- どちらも環境変数未設定なら **本家と同一コードパス**。
- `COST_MV` への追加候補評価は次段。まず seed のみで
  「ヒント無しと同じ出力」を確認する。
- パッチは `tools/apply_x264_hint.py` で適用し、`--revert` で戻せる。

## 4. 検証ゲート

1. **上流ビルド通過**（CLI 含む）。
2. **ヒント無し・dump 無しで本家と MD5 一致**。
3. **ヒント有りで ME 時間削減**（同一出力または BD-rate +0.5% 以内）。
4. **BD-rate PSNR/SSIM +0.5% 以内**（可能なら VMAF）。
5. **GPU 無しで本家と同一出力**。
6. **起動時計測で GPU 経路自動 OFF**。

### 測定コマンド

```sh
# 本家
x264 --input-res 1920x1080 --fps 30 --frames 120 --preset medium --crf 27 \
  -o build/work/base.264 build/work/cockL_1080_120.y4m
# パッチ後（ヒント無し）
X264_MV_HINT= X264_MV_DUMP= ./forks/x264/x264 \
  --input-res 1920x1080 --fps 30 --frames 120 --preset medium --crf 27 \
  -o build/work/patched.264 build/work/cockL_1080_120.y4m
cmp build/work/base.264 build/work/patched.264
```

## 5. GPL

- x264/x265 フォーク全体は **GPLv2/GPLv3**。aeg 本体とは別配布。
- 動的リンク・プロセス分離でも GPL の解釈は法務確認が必要。
- CUDA は SDK をリンクせず、clang で PTX を生成し、
  `libcuda.so` を `dlopen` する（FFmpeg の cuda-llvm 方式）。
- OpenCL は ICD ローダを `dlopen`。GPU カーネルは GPL 側に含める。
- 配布物に GPL 通知とソース提供義務を付ける。

## 6. CI

- GPU 無しランナー:
  - 上流ビルド、`--help`、本家との MD5 一致。
  - ヒント無しで出力一致。
  - `X264_MV_DUMP` のラウンドトリップ。
- GPU ランナー:
  - 1080p 120f `go_gate.py` の base/all_off。
  - ヒント有り/無しの BD-rate（PSNR/SSIM、VMAF は任意）。
  - 起動時計測で GPU OFF の確認。
- すべての数値を `docs/GO_GATE_*.md` に自動生成。

## 7. 次段

1. x264 0.164 の取得・ビルド・パッチ適用。
2. seed ヒントの同一出力確認。
3. `COST_MV` 候補追加版の評価。
4. x265 アダプタ（`Search::singleMotionEstimation`）。
5. GPU 解析エンジン層（aeg backend を再利用）。

## 8. 現時点の制約

- **ソース未取得のため、ビルド・MD5・ME 時間・BD-rate は未測定**。
- `forks/x264/README_ja.md` の手順は、ネットワーク許可後に実行する。
- aeg 既定（197,721 バイト）は不変。
