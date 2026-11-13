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
## 0.1 Update M-9B
x264 0.165.x rev 0480cb0 is built in forks/x264 with MinGW 32bit --disable-asm. Patch: Makefile 1/1, me.c 7/0. Env unset matches upstream MD5. Hint adds a candidate but does not reduce ME time; output +0.9%. ESA call count differs; keyed lookup needed.
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
## M-9E フラグ伝播とヒントモード予備実測

### 実装
- `x264_t` に `aeg_in_encode` / `aeg_enc_frame_idx` を追加。
- `encoder/encoder.c` の `slices_write()`（実際のスライス書き込みスレッド）で
  `aeg_mv_hint_begin_frame(h)` / `end_frame(h)`。
- `aeg_mv_hint.c` は `h->aeg_in_encode` の時だけ dump/hint。
- フレーム番号は `slices_write` 呼び出しごとの単調カウンタ。
- `X264_MV_HINT_MODE=extra|seed|skip|limit:<N>`。
- `skip` は fullpel 探索を飛ばし、subpel refine も飛ばす。
- `X264_MV_HINT_STATS=1` で `frames/hits/misses/applied` を atexit 出力。

### dump の確認
320x192 4f、`--threads 1 --me hex --subme 7`：
- dump 行数 **2,524**
- フレーム番号は 1,2,3（I フレームは ME なし）
- 自己ヒント（同じ dump を extra で注入）で出力一致
- stats: `frames=4 hits=2524 misses=0 applied=2524`

**注意**: `--threads 1` でないと、フレームスレッド間で
`aeg_in_encode` / frame counter が競合し、dump が一部フレームだけになる。
本測定は `--threads 1` で行った。

### 予備測定（1080p 12f cockL、medium/CRF27、3回中央値）

| mode | 時間 | 出力 |
|---|---:|---:|
| base（ヒント無し） | 11.53 s | 70,626 B |
| extra | 10.66 s | 70,626 B |
| seed | 10.79 s | 72,068 B |
| skip（fullpel+subpel 省略） | **10.11 s** | 70,031 B |
| limit:2 | 10.60 s | 72,132 B |
| limit:4 | 10.44 s | 72,118 B |

- 時間は 12f で負荷変動が大きく、base は 9.36–14.52 s。
- `skip` は約 **12% 短縮**、出力は -0.8%（同 QP）。
- `seed` / `limit` は 1–8% 短縮だが、ビットが +2% 前後。
- **BD-rate の全 CRF/4 クリップ測定は未完了**。予備値では skip が
  「+0.5% 以内」を満たす可能性があるが、品質（PSNR/SSIM）未測定。

### ゲート
| ゲート | 結果 |
|---|---|
| 上流ビルド | PASS |
| env 未設定で本家 MD5 一致 | PASS |
| dump 生成 | PASS（--threads 1） |
| 自己ヒント一致 | PASS |
| seed/skip/limit クラッシュ無し | PASS（--threads 1） |
| 時間短縮 | skip 約 12%（12f 予備） |
| BD-rate +0.5% 以内 | **未測定** |
| 既定は本家挙動 | PASS |

### 次段
- `--threads 1` 依存を外すため、`x264_encoder_encode` の
  フレーム起動前に `h->thread[i]` 全部へフラグと frame idx を設定する。
- 1080p 120f × 4 クリップ × CRF23/27/31 の PSNR/SSIM BD-rate。
- skip の品質が悪い場合、limit:N の N スイープでパレート点を取る。
## M-9F BD-rate 実測（途中）

測定: `tools/x264_hint_bd.py`、1080p 120f cockL、medium、CRF27、3回中央値。
ヒントは同一 x264 の hex dump。`--threads 1`。
1 エンコードが約 100–120 s（32bit/no-asm）のため、4クリップ×3CRF×2ヒントの
完全マトリクスは今回完走できなかった。

| mode | 時間 s | 出力 B | PSNR | SSIM | 対 base 速度 |
|---|---:|---:|---:|---:|---:|
| base | 103.618 | 504,911 | 46.3547 | 0.991040 | 1.00x |
| skip | 112.854 | 500,985 | 46.3506 | 0.991072 | **0.92x** |
| seed | 115.714 | 508,201 | 46.3411 | 0.991066 | **0.90x** |
| limit:2 | 114.641 | 509,231 | 46.3523 | 0.991037 | **0.90x** |
| limit:4 | 116.215 | 510,180 | 46.3604 | 0.991049 | **0.89x** |

- 品質はほぼ同等（PSNR 差 <0.02 dB、SSIM 差 <0.00005）。
- skip はビット **-0.8%** だが、時間は **+9%**。
- seed/limit はビット +0.6〜1.0%、時間 +10〜11%。
- **ヒント lookup と候補評価のオーバーヘッドが、探索短縮を上回る**。
- 全モードが base より遅く、時間短縮ゲートは **FAIL**。
- BD-rate は 1 CRF のみのため算出不能（表は参考値）。

### 原因
- hint ファイル（120f で数十万行）のロード/二分探索を各 ME 呼び出しで行う。
- `skip` でも fullpel/subpel の一部を残しており、削減幅が小さい。
- 32bit/no-asm ビルドでは ME 以外の処理が支配的で、ヒントの効果が見えにくい。

### 判定
- 案C（パレート改善）に対して、現行の seed/skip/limit は **有望ではない**。
- 次段は「ヒントを外部ファイルではなく in-memory 配列で渡す」
  「探索開始点の変更だけに限定」「subpel まで完全にスキップ」の再設計が必要。
## M-9G ヒント参照の in-memory 化と limit:2 の実測

### 実装
- `aeg_mv_hint.c`: ヒントを初回だけロードし、open-addressing hash
  （key = frame/mbx/mby/ref/i_pixel）で **O(1) lookup**。
  `getenv` は static キャッシュ。
- `me.c`: `skip` はヒント MV を 1 回 `COST_MV` した後、
  fullpel/subpel を一切行わず `m->mv` / `m->cost` を確定して return。
- `limit:N`: ヒント位置から **マンハッタン半径 N のダイヤ探索**だけを行い、
  hex ループを回さない。
- stats: `skip_calls`, `limit_calls` を追加。
- env 未設定は本家と MD5 一致（1080p 12f）。

追跡ファイル差分:
```
Makefile         1 + / 1 -
common/common.h  2 + / 0 -
encoder/encoder.c 3 + / 0 -
encoder/me.c     37 + / 0 -
```
＋ `encoder/aeg_mv_hint.{c,h}`。

### 640x352 / cockL 240f / CRF27 / medium / `--threads 1`

| mode | 時間中央値 | 出力 |
|---|---:|---:|
| base | 18.31 s | 99,967 B |
| **limit:2** | **16.26 s** | **100,149 B** |
| limit:1 | 16.37 s | 100,343 B |
| limit:4 | 18.38 s | 100,261 B |
| skip | 19.34 s | 129,570 B |

- **limit:2 は 11.2% 高速**、出力 +0.18%。
- limit:1 は 10.6% 高速だが PSNR -0.12 dB。
- skip は遅く、ビット +29.6%。不採用。

### BD-rate（cockL / vtestL、640p 240f、CRF23/27/31）

limit:2 vs base、正 = 悪化。

| clip | PSNR | SSIM |
|---|---:|---:|
| cockL | **+0.176%** | **+0.325%** |
| vtestL | **+0.130%** | **-0.050%** |
| 平均 | **+0.153%** | **+0.138%** |

- **BD-rate +0.5% 以内を満たす**。
- 640p の 4 クリップ CRF27 では、limit:2 が全クリップで base より高速：
  - cockL 18.31→16.26 s（-11.2%）
  - vtestL 12.34→10.82 s（-12.3%）
  - megaL 10.29→8.92 s（-13.3%）
  - treeL 4.99→4.45 s（-10.8%）
  - 出力は -0.7%〜+0.3%。

### 判定
- **640p で base より速いモードは存在する：limit:2**。
- **BD-rate +0.5% 以内（2 クリップの全 CRF で確認）**。
- 案C（1080p+ / medium–slow のパレート改善）に対して、
  `limit:2` が現時点で唯一の有望モード。
- 1080p 120f は 1 回 100 秒のため未測定。次段は 2 クリップ × CRF27 で
  time/bit を確認し、必要なら limit:1/3 をスイープする。

### 注意
- 測定は `--threads 1`。フレームスレッド伝播は未実装。
- ヒントは hex dump（`--me hex --subme 7`）。
- ESA 由来ヒントは未測定。
## M-9H limit:2 確定（1080p 実測・4クリップ BD-rate）

### 1080p 120f / medium / CRF27 / `--threads 1`

交互実行、各 3 回中央値。

| clip | base | limit:2 | 短縮 | base bytes | limit bytes |
|---|---:|---:|---:|---:|---:|
| cockL | 92.10 s | **81.10 s** | **-11.9%** | 504,911 | 502,061 |
| vtestL | 123.97 s | **112.57 s** | **-9.2%** | 823,806 | 824,525 |

- 1080p でも **9–12% の短縮**。
- ビットは cockL -0.56%、vtestL +0.09%。
- 画質（PSNR/SSIM）は 1080p では未測定。

### 640p 4クリップ完全 BD-rate（limit:2 vs base）

240f、CRF23/27/31、PSNR/SSIM はデコード後。正=悪化。

| clip | PSNR | SSIM |
|---|---:|---:|
| cockL | +0.176% | +0.325% |
| vtestL | +0.130% | -0.050% |
| megaL | **-0.579%** | **-0.567%** |
| treeL | **-0.056%** | **-0.110%** |
| **平均** | **-0.082%** | **-0.100%** |

- 4 クリップすべて **+0.5% 以内**。
- 平均は負（改善）。megaL は明確に改善。

### 640p CRF27 時間（4クリップ）

| clip | base | limit:2 | 短縮 |
|---|---:|---:|---:|
| cockL | 18.31 s | 16.26 s | **-11.2%** |
| vtestL | 12.34 s | 10.82 s | **-12.3%** |
| megaL | 10.29 s | 8.92 s | **-13.3%** |
| treeL | 4.99 s | 4.45 s | **-10.8%** |

### 判定

| ゲート | 結果 |
|---|---|
| 1080p で短縮 | **YES（-9.2〜-11.9%）** |
| 640p 4クリップ BD-rate +0.5% 以内 | **YES（平均 -0.082/-0.100%）** |
| 案C（パレート改善） | **limit:2 が候補として確定** |

### 未測定
- `limit:1` / `limit:3` の 4クリップ比較（時間）。
- ESA 由来ヒント（hex のみ）。
- 1080p の PSNR/SSIM。
- フレームスレッド対応（`--threads 1` のみ）。
## M-9I フレームスレッド対応と 1080p 品質

### 実装
- `aeg_mv_hint_set_frame(h, frame_idx)` を追加。
- `x264_encoder_encode()` の実際のフレーム書き込み前に
  `h->thread[0..i_threads]` の全 `x264_t` へ
  `aeg_in_encode=1` と `aeg_enc_frame_idx=frame_idx` を設定。
- `get/post` は `h->fenc->i_frame == aeg_enc_frame_idx` の時だけ動作。
  これで lookahead / 別フレームのスレッドは自動的に除外される。
- `slices_write()` 単位の begin/end は廃止。

差分:
```
Makefile         1 + / 1 -
common/common.h  2 + / 0 -
encoder/encoder.c 5 + / 0 -
encoder/me.c     37 + / 0 -
```
＋ `encoder/aeg_mv_hint.{c,h}`。

### dump（640p 240f、hex）
- `--threads 1`: 974,112 行
- `--threads 4`: **1,498,547 行**（フレーム書き込みの再試行で重複）
- stats: `frames=480`（240 フレーム + 再試行）
- ハッシュは重複キーを上書きするため lookup は一意。

### 640p / CRF27 / `--threads 4`（3回中央値）

| clip | base | limit:2 | 短縮 | base B | limit B |
|---|---:|---:|---:|---:|---:|
| cockL | 13.77 s | **9.78 s** | **-29.0%** | 99,967 | 100,195 |
| vtestL | 10.28 s | **7.53 s** | **-26.7%** | 158,327 | 158,910 |
| megaL | 8.47 s | **6.07 s** | **-28.3%** | 143,071 | 142,860 |
| treeL | 3.49 s | **2.90 s** | **-16.9%** | 168,446 | 168,446 |

- 4 スレッドでも **17–29% 短縮**。`--threads 1` の 11–13% より大きい。
  ME がフレーム並列化されても、ヒントで各フレームの ME が短くなるため。
- 出力差は -0.1%〜+0.4%。

### 1080p 120f / CRF27 / medium / `--threads 1`

| clip | base | limit:2 | 短縮 | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| cockL | 92.10 s | 81.10 s | -11.9% | 46.3548 → 46.3425 | 0.991040 → 0.991038 |
| vtestL | 123.97 s | 112.57 s | -9.2% | 未測定 | 未測定 |

- cockL は **PSNR -0.012 dB / SSIM -0.000002**、ビット -0.56%。
- 1080p の BD-rate 3点は未測定。

### ESA 由来ヒント（640p cockL、threads 4）

- ESA dump: 1,503,957 行、ヒント生成 38.45 s。
- limit:2 時間 **10.55–10.88 s**（hex ヒント 9.78 s より遅い）。
- 出力 **109,862–110,816 B**（hex 100,195 B より +10%）。
- ESA はサブペル 9 の高品質 MV だが、ヒントのキー/範囲変換と
  ME 方式差で hex より悪化。現状は hex ヒントが良い。

### 判定

| ゲート | 結果 |
|---|---|
| フレームスレッドで dump 全フレーム | **PASS（重複あり）** |
| `--threads 4` で短縮 | **PASS（17–29%）** |
| env 未設定で本家 MD5 一致 | **PASS（threads 4 含む）** |
| 1080p 短縮 | **PASS（9–12%）** |
| 1080p 品質 | cockL はほぼ同等。3点 BD 未測定 |
| ESA ヒント | hex より悪化 |

`limit:2` はマルチスレッドでも有効です。次段は dump の重複除去、
1080p の CRF23/31、ESA ヒントのキー正規化です。
## M-9J ヒント生成コストの現実化

### 測定（640p cockL 240f、threads 4、CRF27）

| 項目 | 時間 | 備考 |
|---|---:|---|
| base（ヒント無し） | 13.77 s | 99,967 B |
| hex dump 生成 | 13.8 s | 通常エンコード相当 |
| hex ヒント + limit:2 | 9.78 s | 100,195 B |
| **hex エンドツーエンド** | **約 23.6 s** | **base の 1.71x 遅い** |
| 低解像度（320x176）dump | 10.02 s | 36,678 B |
| 低解像度ヒント ×2 + limit:2 | 9.60 s | **118,781 B（+19%）** |
| **低解像度 E2E** | **19.6 s** | **base の 1.42x 遅い** |
| hex + subme0 dump | 24.5 s | 123,117 B |
| ESA dump | 38.5 s | 99,693 B |

- **CPU でヒントを作る限り、エンドツーエンドで base を下回らない**。
  dump 生成がほぼ通常エンコード 1 回分のコストだから。
- `limit:2` 自体は 9.78 s と速いが、ヒント生成 13.8 s を足すと負ける。
- 低解像度ヒントは生成 10 s でも、MV ×2 の品質が悪く
  ビット +19%。スケールだけでは不足。

### 判定
**GPU（または同等に安い解析）でヒントを生成しない限り、案Cの
エンドツーエンド改善は成立しない。** CPU dump 方式は不採用。

### GPU ヒント生成の設計（M-8B へ）

既存 aeg の CUDA/OpenCL lowres バッチ ME を再利用する。

入力:
- 現在フレームの luma（GPU 常駐なら 0 転送、CPU からなら 1080p で約 2.0 MB）。

処理:
- 1/2 または 1/4 解像度に縮小（GPU 前処理）。
- `lowres_sad_batch` / `lowres_batch_search` で MB ごとの整数 MV を生成。
- 必要なら上位 1 候補のみ。コストは不要（x264 側で再評価）。

出力（ヒントファイル形式）:
```
frame mbx mby mvx mvy cost ref i_pixel
```
- 1080p: 8,160 MB × 8 B = **65 KB/frame**（≤128 KB ゲート内）。
- 640p: 920 MB × 8 B = **7.4 KB/frame**。

転送・同期:
- フレーム単位 1 回のカーネル起動＋1 同期。
- 入力 luma の upload は既存の再構成/ME upload と共有。
- ヒント download は 65 KB/frame のみ。
- ページロック＋非同期コピー。

品質:
- 低解像度 MV ×2 をそのまま使うと品質が落ちるため、
  `limit:2` のダイヤ探索で補正する。
- 現在の CPU 低解像度ヒントは +19% ビット。GPU では
  (a) 1/2 解像度で `subme` 相当の小数探索、
  (b) 上位 2 候補、
  (c) 原画との SAD 再評価
  を GPU 内で行い、品質を改善する必要がある。

### 分割案
- **M-9J1**: GPU 低解像度ヒント生成（1/2、整数 MV、1 候補）。
- **M-9J2**: ヒント品質改善（小数探索・上位 2 候補・SAD 再評価）。
- **M-9J3**: エンドツーエンド測定（GPU ヒント生成＋limit:2 vs base）。
- CPU dump 方式は M-9J で不採用と確定。
## M-9J1 GPU ヒント生成ツール（試作）

### 実装
`tools/gpu_hints.c`（244 行、`make build/gpu_hints`）:
- YUV420p → luma を 1/2 解像度へ縮小。
- aeg backend の `lowres_batch_search` で 8x8 ブロック（= full-res 16x16 MB）
  の整数 ME。1 フレーム 1 呼び出し。
- full-res luma 上で `me_batch` により ±2 整数リファイン。
  1080p は帯分割（4 MB 行ずつ）。
- 出力は x264 dump と同一の 8 列。MV は quarter-pel（×4）、`ref=0`、
  `i_pixel=1`。
- `-b cpu` で CPU フォールバック。

### 640p / 240f / threads 4 / CRF27

| clip | hint 生成 | limit:2 | base | E2E (生成+limit) | 判定 |
|---|---:|---:|---:|---:|---|
| cockL | 5.5 s | 8.6 s | 13.1 s | 14.1 s | 観測値は +1.0 s 負け |
| vtestL | 2.2 s | 5.9 s | 9.7 s | 8.1 s | **-1.6 s 勝ち** |
| megaL | 2.0 s | 4.9 s | 8.2 s | 6.9 s | **-1.3 s 勝ち** |
| treeL | 1.5 s | 2.5 s | 3.5 s | 4.0 s | +0.5 s 負け |

- 生成時間は `/mnt/q` へのテキスト書き込みを含む。WSL DrvFs の
  書き込みが約 2 MB/s になり、cockL では 4.6 s がファイル出力に費やされる。
- `/dev/null` 出力時の計算時間は 640p で **約 1.0 s（4.2 ms/frame）**。
  目標 2 ms/frame は未達。
- 1080p は `cuMemcpyHtoDAsync` で SIGSEGV（WSL CUDA、ページャブル
  2 MB 超コピー）。**1080p は未完成**。

### 品質（640p、GPU ヒント + limit:2）

| clip | PSNR BD-rate | SSIM BD-rate |
|---|---:|---:|
| cockL | 0.0% | 0.0% |
| vtestL | -0.06% | -0.21% |
| megaL | **+0.76%** | **+1.08%** |
| treeL | -0.59% | -4.35% |
| 平均 | +0.03% | -0.87% |

- 出力バイト数は全クリップ・全 CRF で base と**完全に同一**。
- cockL は PSNR/SSIM も完全一致。vtestL/treeL は同等。
- **megaL は +0.5% を超えて悪化**。平均では +0.5% 以内だが、クリップ
  単位のゲートは満たさない。

### 判定
- **640p の一部クリップでは E2E 勝ち**（vtestL -1.6 s、megaL -1.3 s）。
- cockL/treeL はファイル出力の遅さで負け。計算のみなら cockL は
  9.6 s vs 13.1 s で勝ち。
- **BD-rate +0.5% 以内は megaL が FAIL**。
- 1080p はクラッシュで未測定。
- したがって「案C 実証完了」には未達。不足分:
  - 1080p の pinned staging（2 MB 超コピー）
  - megaL の品質 +0.76% → 0% へ（候補 2 本 or ±1 リファイン）
  - 640p 計算 4.2 ms → 2 ms（帯分割の廃止、full-res リファイン省略）
  - ファイル出力 4.6 s → 0.3 s（バイナリ中間 or ローカル一時ファイル）

### 分割案
- **M-9J1a**: 1080p pinned staging と帯分割の修正。
- **M-9J1b**: megaL 品質改善（候補 2 本、full-res ±1、SAD 再評価）。
- **M-9J1c**: 出力のバイナリ化＋変換ツールで E2E を確定。
- **M-9J1d**: 640p 2 ms/frame 化（帯分割廃止、リファイン範囲削減）。
## M-9J2 バイナリ出力・1080p・megaL 品質

### 1. バイナリ出力（実装済み）

- `tools/gpu_hints.c`: 出力先が `.bin` なら 12 バイト固定長で書く。
  ```
  uint16 frame, mbx, mby; int16 mvx, mvy; uint8 ref, pix;
  ```
- `forks/x264/encoder/aeg_mv_hint.c`: `X264_MV_HINT` の拡張子が `.bin`
  ならバイナリとしてロード。`.txt` は従来のテキスト。
- 640p 240f の出力サイズ: 4.76 MB（テキスト）→ **2.53 MB**。
- 640p cockL の生成時間: **2.28 s**（テキスト 5.5 s）。
  - 目標 ≤1.3 s は未達。内訳は計算約 1.0 s、書き込み約 1.2 s。
- 1080p は生成前に `me_batch` で SIGSEGV のため未測定。

### 2. ローダの回帰（未解決）

バイナリ／テキストともロード自体は全件成功するが、
`limit:2` のヒットが **1,801** に落ちた（以前は約 1.1M）。

```
aeg_mv_hint: loaded 211200 text entries
aeg_mv_hint: frames=480 hits=1801 misses=1494978 applied=1801
```

- 出力ビット数は base と同一（99,967 B）だが、ヒントがほぼ効いていない。
- 原因候補: フレーム番号と `aeg_enc_frame_idx` の対応
  （`fenc->i_frame` / `begin_frame` / フレームスレッド）。
- このため M-9J2 の BD-rate 測定は無効。M-9J1 のテキスト測定値
  （megaL +0.76%）が最後の有効値。

### 3. 1080p crash（未解決）

`cuMemcpyHtoDAsync` の 2 MB 超ページャブルコピーで SIGSEGV。
帯分割（4 MB 行、cbytes 約 254 KB）でも再現。
pinned staging は未実装。

### 4. megaL 品質（未実施）

バイナリ・ローダが未復旧のため、上位 2 候補・±1 リファインの
品質測定は未実施。

### 判定

| 項目 | 結果 |
|---|---|
| バイナリ出力 | 実装済み、2.53 MB / 2.28 s |
| 目標 ≤1.3 s | **FAIL（2.28 s）** |
| バイナリ・ローダ | **FAIL（hits 1,801）** |
| 1080p crash | **未解決** |
| megaL +0.5% | **未測定**（M-9J1 は +0.76% FAIL） |
| 案C 実証 | **未達** |

### 次分割
- **M-9J2a**: ローダのフレーム対応を修復（ヒント全件ヒット）。
- **M-9J2b**: pinned staging と 1080p の段階測定。
- **M-9J2c**: 上位 2 候補・±1 リファインで megaL 品質。
- **M-9J2d**: 出力 1.2 s → 0.2 s（バッファ拡大・直接 write）。
## M-9J2a ヒントローダのフレーム対応修復

### 原因
`x264_me_t.i_pixel` の 16x16 値は **3**（`PIXEL_16x16`）だが、
`gpu_hints` が **1** を書いていた。キーは
`(frame, mbx, mby, ref, i_pixel)` なので、全ヒントが不一致となり
hits が 1,801 に落ちていた。

### 修正
- `tools/gpu_hints.c`: `i_pixel` を **3** に変更（テキスト／バイナリ両方）。
- これでテキスト経路とバイナリ経路の hits がほぼ一致:
  - 60f: text 60,860 / bin 60,869（差は重複キーの丸めのみ）
- 240f / threads 4 / CRF27:
  - **hits 529,237 / misses 967,994**（1,801 から回復）
  - 出力 119,811–120,655 B（base 99,967 B より **+20%**）
  - `limit:2` 時間 **7.78–7.99 s**（base 13.1 s）

### 残る問題
- hits は 1.1M ではなく **529k**。`gpu_hints` のフレーム番号と
  x264 の `fenc->i_frame` が B ピラミッドで一致しない可能性。
  ただし全キーが不一致ではない。
- 出力が +20% に悪化。ヒントが実際に適用されるようになったため、
  `gpu_hints` の MV 品質（低解像度×2＋±2）が不足している。
  M-9J1 の「99967 B」はヒント未適用（pix=1）の値だった。
- 生成時間は **2.25 s** のまま。1 回の `fwrite` にまとめても
  変化せず、`/mnt/q` の書き込みが約 1.2 s を占める。

### 判定
| 項目 | 結果 |
|---|---|
| テキスト／バイナリ parity | **PASS**（ほぼ一致） |
| hits 回復 | **一部 PASS（1,801 → 529k）** |
| 目標 1.1M | **FAIL** |
| limit:2 時間 | 7.8–8.0 s（base 13.1 s） |
| 出力 | **+20%（品質悪化）** |
| 生成 ≤1.3 s | **FAIL（2.25 s）** |

### 次
- M-9J2b: frame 番号の完全一致（B ピラミッド）。
- M-9J2c: 上位 2 候補＋±1 リファインで +20% を解消。
- M-9J2d: 出力 1.2 s の削減（ローカル一時ファイル）。
