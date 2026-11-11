## 9. M-9B 実測：x264 fork ビルドとヒント注入

対象: `forks/x264` x264 0.165.x rev 0480cb0。
ビルド: `--host=i686-pc-mingw32 --disable-asm`、32bit、`AR=ar rc `。
絶対速度は 32bit/no-asm のため本家 x64 と比較しない。同一ビルド内の A/B のみ有効。

### 素のビルド基準（1080p 120f、CRF27、各3回中央値）

| clip | medium | slow |
|---|---:|---:|
| cockL | 10.61 s | 16.55 s |
| vtestL | 10.65 s | 16.92 s |
| megaL | 10.63 s | 16.75 s |
| treeL | 10.65 s | 16.60 s |

出力は全クリップ同一設定で安定（例 treeL medium 3,491,117 B）。

### パッチ差分

`git -C forks/x264 diff --numstat`:

```
1  1  Makefile
7  0  encoder/me.c
```

- `encoder/aeg_mv_hint.{c,h}` を追加（未追跡）。
- `x264_me_search_ref()` に hint 候補評価 5 行、dump 1 行。
- Makefile に 1 ソース追加。
- **既定（env 未設定）で本家と MD5 一致**（1080p 12f cockL）。

### ヒント注入の実測（1080p 12f、cockL、medium/hex/subme7）

| 条件 | 時間中央値 | 出力 |
|---|---:|---:|
| ヒント無し | 7.25 s | 104,701 B |
| ヒント有り（hex dump を fullpel 変換） | 7.30 s | 105,667 B |

- **ME 時間は減らない**。ヒントは探索開始点ではなく、予測候補の後に
  `COST_MV` で 1 回追加評価するだけなので、hex 探索はそのまま走る。
- 出力は +0.9% 大きい。seed が探索結果と一致しないため。
- 生の dump MV（quarter-pel）をそのまま `COST_MV` に渡すと範囲外アクセスで
  segfault した。fullpel 変換とクランプが必要。
- 320x192 の同一設定では、dump を 2 列に変換して戻すと **出力一致**。
  これは「ヒントが探索結果と完全一致し、追加評価が最良を変えない」条件。

### ESA dump の問題

- 320x192 4f: hex dump 4444 行、ESA dump 4496 行。
- `x264_me_search_ref()` の呼び出し回数が ME 方式で異なるため、
  2 列の順序ファイルをそのまま hex に注入すると **対応がずれる**。
- 対策: dump に `frame, mbx, mby, ref, i_pixel` を追加し、
  キーでヒントを引く形式にする（今回の dump は 8 列化済み）。
- ただし現行の `X264_MV_HINT` は 2 列の順次読みなので、
  8 列キー形式の lookup は未実装。

### ゲート判定

| ゲート | 結果 |
|---|---|
| 上流ビルド | **PASS** |
| env 未設定で本家 MD5 一致 | **PASS**（1080p 12f） |
| ヒントで ME 時間削減 | **FAIL**（7.25→7.30 s） |
| BD-rate +0.5% 以内 | **FAIL の可能性**（同 QP で +0.9% サイズ） |
| 既定は本家挙動 | **PASS** |

**結論**: 現行の「候補を 1 回追加評価する」実装では速度改善はない。
次段は「ヒントを探索の開始点にする」「ヒント周辺だけ探索する」
「キー付きヒント lookup」の 3 点が必要。
Updated medians from x264_bare.csv: cockL 9.01/11.47, vtestL 10.19/11.19, megaL 9.40/11.22, treeL 10.58/16.60 (medium/slow).
