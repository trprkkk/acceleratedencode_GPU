# x264 フォーク作業ディレクトリ

## 状態（run 58 時点）

**未初期化**。外部からの `git clone` がサンドボックスに拒否されたため、
x264 のソースツリーはまだ置かれていない。以下は、ソースを配置した後に
そのまま実行できる手順と、最小差分パッチの方針。

## 対象

- x264 0.164 系（既存バイナリは 0.152.2854。まず 0.164 系でビルドし、
  0.152 で測定した上限との差を確認する）。
- GPLv2。フォーク全体が GPL。aeg（MIT/BSD 系の自前実装）とは別配布にする。

## 取得（ユーザーがネットワークを許可した環境で）

```sh
git clone --depth 1 --branch stable https://code.videolan.org/videolan/x264.git forks/x264
# または GitHub mirror
# git clone --depth 1 https://github.com/mirror/x264.git forks/x264
cd forks/x264
git rev-parse HEAD > ../X264_REVISION
```

`X264_REVISION` にコミットを記録する。

## ビルド

```sh
./configure --enable-static --disable-cli=no --enable-pic
make -j"$(nproc)"
./x264 --version
```

CLI 込みでビルドする。`--disable-opencl` は使わない（GPU 側は別レイヤ）。

## パッチ適用

```sh
python3 ../../tools/apply_x264_hint.py --src .
make -j"$(nproc)"
```

パッチは環境変数でゲートする。既定は本家挙動。

- `X264_MV_HINT=<file>`: ME 開始前に `m->mv` を seed として設定。
- `X264_MV_DUMP=<file>`: `x264_me_search_ref()` の後に
  `mv_x mv_y cost i_pixel` を 1 行追記。

`encoder/aeg_mv_hint.c` / `.h` を追加し、`encoder/me.c` の
`x264_me_search_ref()` の入口と出口に 2 行ずつ挿入するだけの最小差分。

## 同一出力の確認

```sh
# 本家
./x264 --input-res 1920x1080 --fps 30 --frames 120 --preset medium \
  --crf 27 -o /tmp/base.264 ../../build/work/cockL_1080_120.y4m
# パッチ後、ヒント無し
env -u X264_MV_HINT -u X264_MV_DUMP ./x264 ... -o /tmp/patched.264 ...
cmp /tmp/base.264 /tmp/patched.264
```

ヒント無し・dump 無しで **MD5 一致**することを必須ゲートにする。

## ヒントの生成

1. `--me esa` で `X264_MV_DUMP` を有効にしてダンプ。
2. 同一解像度・同一フレームのヒントとして `X264_MV_HINT` に渡す。
3. seed のみ（探索範囲を変えない）なら、ヒント無しと同じ最良 MV に
   収束することを確認する。収束しない場合は候補評価の追加が必要。

## 注意

- パッチは x264 の `x264_me_t` の `mv[2]` / `cost` / `i_pixel` を前提にする。
  0.164 系でフィールド名が異なる場合は `aeg_mv_hint.c` を修正する。
- `x264_me_search_ref()` の出口 brace 検出は簡易的な brace カウント。
  適用後に `git diff` で挿入位置を必ず確認する。
- このディレクトリの内容は GPL。aeg 本体とはライセンスを分ける。
