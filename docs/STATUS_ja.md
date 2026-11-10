# 現状ステータス（日本語）

最終更新: run 50 相当。ブランチ `feature/gpu-hybrid-m3-m4`。
数値はすべてこのリポジトリの `build/aegenc` と FFmpeg 3.4.11 で実測したもの。
負荷依存の速度は「同一条件 3 回の中央値」で、環境負荷の注記がある。

## 機能マトリクス

| 機能 | 状態 | 根拠・実測 |
|---|---|---|
| batched lowres（`AEG_LA_BATCH`） | **既定 ON** | 1080p 12f が 202,923 → 197,721 バイト。速度は既定 1.66–2.0 s 帯（負荷依存） |
| implicit weighted B（`--weightb`） | **既定 ON** | CQP BD-rate off 比 **-1.72% PSNR / -2.69% SSIM**、CRF **-1.64% / -3.20%**（4 クリップ平均） |
| open-GOP（`--open-gop`） | opt-in | CPU/CUDA バイト一致、FFmpeg 全数デコード確認 |
| CQM（`--cqm flat\|jvt\|file`） | opt-in | flat 既定。jvt は flat 比 **+7.10% PSNR / +2.13% SSIM**（悪化）。`--dump-recon` とデコード一致 |
| explicit weighted B（`--weightb explicit`） | opt-in | 安全弁付き。CQP **+0.22% PSNR / 0.00% SSIM**、CRF **+0.27% / +0.27%**。通常クリップはほぼ恒等 |
| psy-rd（`--psy-rd`） | **opt-in 凍結** | CRF 最良 mode1 で平均 **+0.20/+0.60/+2.52% PSNR**（0.5/1.0/2.0）、SSIM **+0.52/-1.68/-4.08%**。改善基準未達 |
| 固定小数点 mbtree（`AEG_MBT_FIXED`） | **opt-in 凍結** | 3 バックエンド一致は達成。品質ゲート不合格（参照スロット集合の不一致） |
| `AEG_REC_DEV`（device DPB 常駐） | opt-in | 1080p で **30.16 MB / 12 uploads / 0 download = 2.51 MB/pic**、0.67 calls/pic。CPU/CUDA/OpenCL 一致 |
| `AEG_REC_GPU`（device 再構成） | **未実装** | probe のみ。full source+recon 5.98 MiB/pic、luma-only 3.98 MiB/pic、levels+modes 6.23 MiB/pic。M-1 の 4 MB 超過 |
| `AEG_DEBLOCK_PERSIST`（CPU 参照） | opt-in | ラスタ順 persistent スケジューラ。既定とバイト一致。device 版は M-1.2d-R 待ち |
| `--mux mp4\|mkv\|ts` | opt-in | MP4/MKV/TS の 8bit/10bit を FFmpeg 全数デコード。raw 出力は不変 |
| lossless（`--lossless`） | **opt-in（8bit CAVLC）** | I_PCM 固定。4 クリップで入力 YUV とデコードの MD5 一致。10bit/CABAC は次段 |
| 12bit（`--bit-depth 12`） | **experimental（既定無効）** | obj12/API/SPS profile 244 は実装。12bit 4:2:0 は FFprobe が `yuv420p12le` と認識するが、エンコードで heap corruption / CABAC MB エラーが残る |
| 4:2:2 / 4:4:4 | **未実装** | 入力変換と符号化は 4:2:0 のみ |
| HEVC / x265 コア | **未実装** | 本リポジトリは H.264 のみ |
| `--cqm jvt` | opt-in | 再構成一致。品質は flat より悪い |
| `--mux` 10bit | opt-in | MP4/MKV/TS すべて FFmpeg でデコード可 |
| open-GOP + implicit weighted B | opt-in 併用可 | CPU/CUDA 一致 |

## 既定構成の基準値

1080p（1920x1080）12 フレーム、`-p 2 -q 27 --bframes 3 -g 12`、CUDA:

| 構成 | 出力 | 備考 |
|---|---:|---|
| 既定（implicit weighted B） | **197,721 バイト** | 通常 GOP |
| `--weightb 0` | **197,721 バイト** | 旧既定とバイト一致 |
| `AEG_LA_BATCH=0 --weightb 0` | **202,923 バイト** | 旧 scalar lowres 経路 |

速度は環境負荷で大きく変動する。実測は **1.7〜2.0 s** 帯（3 回中央値）。
同じ 12f でも負荷が高いと 2.1 s 前後になる。

## 既知の制限（1 行ずつ）

- **4:2:2 / 4:4:4**: 4:2:0 のみ実装。
- **12bit**: 12bit ビルドは experimental。encode の heap corruption / CABAC エラーが未解決。
- **lossless**: 8bit CAVLC の I_PCM のみ。10bit / CABAC は未対応。CQM / weighted B / open-GOP との併用はエラー。
- **device 再構成（M-1.2d-R）**: source/recon/levels の転送が 4 MB/pic を超え、device 常駐が前提。
- **クロスピクチャ先読み（M-1.1）**: frame_prologue の状態依存で保留。
- **psy-trellis（M-5d2）**: psy-rd の凍結に依存して未着手。
- **固定小数点 mbtree**: opt-in 凍結。品質ゲート未達。
- **explicit weighted B**: 通常クリップでは安全弁で恒等予測になり、改善は限定的。
- **CQM jvt**: 仕様上は通るが、この実装では flat より BD-rate が悪い。
- **device deblock（M-4a2）**: M-1.2d-R 待ち。
- **コンテナ**: `--mux` は ffmpeg 必須。ES の再取り出しは avcC 再配置で raw とバイト一致しない。

## 関連資料

- コマンド集: `docs/COMMANDS_ja.md`
- 方針・ロードマップ: `docs/DESIGN_POLICY_ja.md`
- 進捗・設計: `docs/PROGRESS_ja.md`, `docs/DESIGN_ja.md`
- 監査: `AUDIT_REPORT_ja.md`, `GPU_REVIEW_ja.md`
