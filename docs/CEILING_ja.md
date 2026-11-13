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
## 6. 探索/RDO/先読みの寄与（M-8A2）

設定切替のみ（改造なし）。各点 3 回中央値。CSV/ログは
`build/work/contrib_x265_*.csv`, `contrib_x264_*.csv`。
環境負荷が高く、640p では固定オーバーヘッド、1080p は 12f 短尺のため、
絶対値より**同一実行内の比**を見る。

### x265 640x352 / cockL 240f / CRF27

| preset | base | subme0 | rd1 | no-cutree+LA5 | all_off | all_off倍率 |
|---|---:|---:|---:|---:|---:|---:|
| medium | 1.043 | 1.014 | 0.870 | 1.420 | 0.934 | **1.12x** |
| slow | 1.790 | 1.615 | 0.948 | 2.924 | 1.300 | **1.38x** |

- `--me full` は 11.8–21.6 s（base の 8–12 倍遅い）。
- `--me dia` は base より遅い（0.85x）。ME 単独は支配的でない。
- `--no-cutree` / `--rc-lookahead 5` はこの環境では逆に遅く、ノイズが大きい。
- `rd1` が最も効き、medium 1.20x / slow 1.89x。

### x265 1920x1080 / 12f / CRF27

| preset | base | subme0 | rd1 | no-cutree+LA5 | all_off | all_off倍率 |
|---|---:|---:|---:|---:|---:|---:|
| medium | 0.558 | 0.557 | 0.523 | 0.548 | 0.489 | **1.14x** |
| slow | 0.820 | 0.720 | 0.634 | 0.828 | 0.571 | **1.44x** |

- ME 単独（subme0）は medium 0%、slow 14%。
- RDO（rd1）は medium 6.7%、slow 29%。
- **合計を全部切っても 1.14–1.44x**。x265 1.5–2x は届かない。

### x264 640x352 / cockL 240f / CRF27

| preset | base | subme0 | trellis0 | no-mbtree | no-LA | all_off | 倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 0.686 | 0.658 | 0.682 | 0.705 | 0.709 | 0.642 | **1.07x** |
| slow | 0.728 | 0.598 | 0.715 | 0.713 | 0.731 | 0.628 | **1.16x** |

- `--me esa` は 0.90–0.96x（遅い）。`dia` はほぼ同等。
- 640p は固定オーバーヘッドが大きく、上限倍率は低い。

### x264 1920x1080 / 12f / CRF27

| preset | base | subme0 | trellis0 | no-mbtree | no-LA | all_off | 倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 0.388 | 0.307 | 0.385 | 0.325 | 0.331 | 0.267 | **1.45x** |
| slow | 0.428 | 0.293 | 0.371 | 0.348 | 0.356 | 0.279 | **1.54x** |

- subme0 は medium 26.5%、slow 46%。
- trellis0 は 15%、MB-tree 23%、lookahead 20% 程度。
- **x264 slow の all_off は 1.54x**。1.2–1.5x 目標は 1080p では上限内。

### 再計算した上限判定

「ME＋先読み＋RDO をすべて無料にした」all_off を上限とみなす。

| encoder/preset | 640p 上限 | 1080p 上限 | 目標 | 判定 |
|---|---:|---:|---|---|
| x265 medium | 1.12x | 1.14x | 1.5–2x | **NO** |
| x265 slow | 1.38x | 1.44x | 1.5–2x | **NO（近い）** |
| x264 medium | 1.07x | 1.45x | 1.2–1.5x | **1080p は YES、640p は NO** |
| x264 slow | 1.16x | 1.54x | 1.2–1.5x | **1080p は YES、640p は NO** |

結論:
- 1080p では x264 1.2–1.5x は上限内。x265 1.5–2x は all_off でも届かない。
- 640p は固定オーバーヘッドのため、探索を無料にしても 1.1–1.4x 止まり。
- x265 で 1.5x を狙うには、RDO 自体の短縮（枝刈り・候補削減）が必須。
- これは「GPU が無限に速い」仮定でも同じ。GPU 実装だけでは目標に届かない。
## 8. 1080p 120f の再測定（固定費の除去）

1080p 12f は固定オーバーヘッドが過大だった。cockL を 1920x1080 に
スケールした 120f で、base と all_off を各 3 回中央値で再測定。

| encoder | preset | base s | all_off s | 上限倍率 |
|---|---|---:|---:|---:|
| x265 | medium | 3.975 | 3.133 | **1.27x** |
| x265 | slow | 7.223 | 3.291 | **2.20x** |
| x264 | medium | 3.043 | 2.401 | **1.27x** |
| x264 | slow | 3.655 | 2.249 | **1.63x** |

- 12f の x265 slow 1.44x から、120f では **2.20x** に上昇。
  固定費が薄まり、RDO/ME の寄与が正しく出る。
- x265 medium は 1.27x のまま。medium では探索より他工程が支配的。
- x264 slow は 1.54x（12f）→ **1.63x**。medium は 1.45x → 1.27x
  （短尺の固定費が除かれて低下）。
- `all_off` は ME/RDO/先読みを全部切った品質非等価の上限。
  したがって、この倍率をそのまま製品速度として約束できない。

### 更新した判定
| encoder/preset | 1080p 12f | 1080p 120f | 目標 | 判定 |
|---|---:|---:|---|---|
| x265 medium | 1.14x | 1.27x | 1.5–2x | **NO** |
| x265 slow | 1.44x | **2.20x** | 1.5–2x | **YES（長尺 slow）** |
| x264 medium | 1.45x | 1.27x | 1.2–1.5x | **YES** |
| x264 slow | 1.54x | **1.63x** | 1.2–1.5x | **YES（上限は目標超え）** |

**重要な修正**: 12f の結果だけから「x265 1.5–2x は原理的に不可」と
結論するのは早計だった。長尺 1080p の slow では 2.2x の余地がある。
一方、medium は x265/x264 とも 1.27x 前後で、1.5x には届かない。
## 9. M-8B1 GO ゲート実測（1080p 120f、長尺4クリップ）

測定: `tools/go_gate.py`。base と all_off を **交互順序**で各 3 回、
中央値の比を採用。絶対時間は負荷依存のため、同一実行内の比を主指標にする。
詳細は `docs/GO_GATE_x264.md`, `docs/GO_GATE_x265.md`。

| encoder | preset | CRF | 平均 | 最悪 | ゲート |
|---|---|---:|---:|---:|---|
| x264 | medium | 23 | 1.43x | 1.36x | ≥1.2 PASS |
| x264 | medium | 27 | 1.39x | 1.35x | ≥1.2 PASS |
| x264 | medium | 31 | 1.42x | 1.29x | ≥1.2 PASS |
| x264 | slow | 23 | 1.76x | 1.55x | ≥1.4 PASS |
| x264 | slow | 27 | 1.73x | 1.51x | ≥1.4 PASS |
| x264 | slow | 31 | 1.67x | 1.44x | ≥1.4 PASS |
| x265 | medium | 23 | 1.37x | 1.29x | ≥1.15 PASS |
| x265 | medium | 27 | 1.47x | 1.37x | ≥1.15 PASS |
| x265 | medium | 31 | 1.36x | 1.15x | ≥1.15 PASS（境界） |
| x265 | slow | 23 | 2.09x | 1.48x | ≥1.5 **FAIL（最悪）** |
| x265 | slow | 27 | 2.22x | 2.03x | ≥1.5 PASS |
| x265 | slow | 31 | 2.16x | 1.97x | ≥1.5 PASS |

- x264 は medium/slow とも全 CRF でゲート通過。
- x265 は slow の平均が 2.09–2.22x だが、CRF23 の vtestL が 1.48x で
  「各クリップ最悪値 ≥1.5」をわずかに割る。
- x265 medium は平均 1.36–1.47x。1.5x は medium では届かない。
- したがって **x265 は slow 限定で 1.5–2x**、medium は 1.3–1.45x が実測上限。

### 最終判定
- **案A（x264 1.4–1.5x、x265 1.3–1.4x）**: x264 は通過。x265 medium も近いが、
  slow の 1.5x を説明できない。
- **案B（速度同等・BD-rate -3〜-5%）**: 速度上限とは独立。GPU を画質に振るなら
  引き続き有力。
- **案C（パレート改善、1080p+ / medium–slow）**: 実測と最も整合。
  x264 medium/slow、x265 medium/slow の各点で上限が取れている。

**推奨**: 案C を採用。x264 は 1.2–1.5x、x265 は medium 1.15–1.45x /
slow 1.5–2.2x を対象に、BD-rate 非悪化を条件とする。
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
