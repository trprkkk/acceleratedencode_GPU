# x264 フォーク（M-9B）

## 状態

**初期化・ビルド済み**。x264 0.165.x rev `0480cb0` が `forks/x264/` にあり、
MinGW 32bit / `--disable-asm` で `x264.exe` をビルド済み。

## ビルド

```sh
cd forks/x264
CC=gcc ./configure --host=i686-pc-mingw32 --disable-asm --enable-static --enable-pic
mingw32-make -j4 "AR=ar rc " RANLIB=ranlib
./x264.exe --version
```

## パッチ

`tools/apply_x264_hint.py` または手動で以下を適用済み：

- `encoder/me.c` に `aeg_mv_hint_get()` の候補評価と `aeg_mv_hint_post()` の dump。
- `encoder/aeg_mv_hint.{c,h}` を追加。
- `Makefile` の SRCS に `encoder/aeg_mv_hint.c` を追加。

`git diff --numstat` は Makefile 1/1、me.c 7/0。既定（env 未設定）は
本家と MD5 一致。

## 実測

- 素の 1080p 120f: medium 10.6s / slow 16.6s（32bit no-asm）。
- ヒント有無（1080p 12f medium）: 7.25s vs 7.30s、出力 104,701 vs 105,667 B。
- 現行のヒントは探索開始点ではなく追加候補評価なので、ME 時間は減らない。
- ESA dump は呼び出し数が hex と異なるため、キー付き lookup が必要。

詳細: `docs/FORK_PLAN_ja.md`, `docs/X264_FORK_RESULT_ja.md`。
