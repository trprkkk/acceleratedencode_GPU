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
