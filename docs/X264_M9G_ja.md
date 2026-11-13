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
