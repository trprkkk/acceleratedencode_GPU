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
