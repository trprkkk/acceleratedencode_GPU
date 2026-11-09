# 進捗記録（H.264 完成まで）

最終更新: 2026-11-08 / ブランチ `arena/589d1e45-acceleratedencode-gpu`
このファイルは**作業を再開するための正本**。コンテキストが切れたら、まずここを読む。

---

## 0. 完了条件（H.264 版）

`docs/DESIGN_ja.md` の G1〜G4 を H.264 に限定したもの。**すべて実測で確認するまで完了にしない。**

| ID | 条件 | 測定方法 | 状態 |
|---|---|---|---|
| H-Q1 | BD-rate ≤ +1.0 %（x264 veryslow 比、CQP・PSNR、14 クリップ） | `tools/bdrate.py` | 未達（p3/p4 で +0.1〜+0.8 %） |
| H-Q2 | BD-rate ≤ +5.0 %（x264 veryslow 比、CRF・長尺） | 同上（CRF モード） | 未達（+11.8 %） |
| H-Q3 | VMAF で x264 veryslow 比 −1.0 以内 | `tools/`（未整備） | 未測定 |
| H-F1 | 4:2:0/4:2:2/4:4:4、8/10/12bit、lossless | 適合試験 | 4:2:0 8/10bit のみ |
| H-F2 | CQM / zones / qpfile / open-GOP / ROI / intra-refresh / slice-max-* | 適合試験 | qpfile・zones・ROI・intra-refresh・slice-max-mbs/size・open-GOP |
| H-F3 | psy-rd / psy-trellis / AQ モード（auto,variance,dark） | BD-rate + 目視 | AQ モード 1/2/3 あり、psy 無し |
| H-F4 | weighted B / B16 / ref16 / Blu-ray 制約 / HDR SEI | 適合試験 | 一部（HDR10 static SEI あり） |
| H-F5 | libavcodec wrapper + MP4/MKV/TS | FFmpeg から使用、mux/demux | 無し |
| H-S1 | 1080p preset4 が x264 veryslow の ≥1.2 倍 | `--bench` | 未達 |
| H-S2 | 1080p preset2 が x264 medium の ≥1.0 倍 | `--bench` | 未達 |
| H-S3 | **GPU 併用で CPU 単独の ≥1.35 倍**（同一ビットストリーム） | `--bench` | **未達（現状 1.0 倍、以前は 0.4 倍）** |
| H-C1 | FFmpeg/JM で再構成一致（全ケース） | `make test` + JM | FFmpeg 3.4 で 148 ケース一致 |
| H-C2 | HW デコーダ 4 種・ブラウザ・コンテナ | 手動マトリクス | 未実施 |

---

## 1. タスクボード

記号: `[ ]` 未着手 / `[~]` 進行中 / `[x]` 完了（テスト済み）

### M-0 配置と正しさの土台 — **完了**
- [x] GCC 7 / clang<12 でビルド（`_mm_storeu_si32` シム）
- [x] PoCL（CPU デバイス）を自動配置から除外（`is_gpu` / `auto_ok`）
- [x] `-s stage=auto` を「auto」の意味に修正
- [x] 係数ステージの自動配置を撤去（実測 2.9〜21 倍遅い）
- [x] `--bench` の bitrate 式バグ
- [x] `tests/dev_bench.c` + `make dev_bench`（性能の再現手段）
- [x] `docs/DESIGN_ja.md`（設計とマイルストーン）

### M-2 探索の一本化＋GPU 主導 — **完了（サブペルはバッチのまま）**
- [x] M-2a: 全画面ラウンド方式（形状別）＋ `aeg_me_refine_around()` — **採用せず**（往復が多すぎる）
- [x] **M-2b: デバイス側完結の整数探索（`me_batch`）**
  - `aeg_be_ops.me_batch`：1 ブロック = 1 ワークアイテム、1 ピクチャ 1 呼び出し
  - CPU 参照（`cpu_me_batch`）と CUDA/OpenCL カーネル（`kernels.h` の `me_search`）が同一の探索
  - `tests/backend_check.c` に 8x8/16x16 の一致・「本当に最小か」・パディング内ブロック・1080 行を追加（**348 チェック**）
- [x] **M-2c（部分）: サブペルも全画面バッチ**。整数は `me_batch` 1 回、半ペル/4 分の 1 ペルは `satd_batch` の全画面ラウンド。8x8 形状のサブペルは既定オフ（既知の 8x8 パディング差のため）。カーネル内への融合は M-1 の常駐化と一緒に行う。
- [x] **M-2d: 品質**。`aeg_me_refine_around()` がラウンドのベクトルの周囲だけを MV 予測子込みで精密化する。既定 GPU 経路は `-b cpu` と比較して 1080p 12 枚で **202,923 vs 204,071 バイト（-0.56%）**、PSNR 49.48 vs 49.45 dB。
- [x] **M-2e: B ピクチャ対応**。`me_seed_frame_b()` が list 0/1 をそれぞれ全画面探索し、`analyse_mb_b()` は各 list を局所精密化するだけ。B は既定 `--bframes 3` の 3/4 を占めるため、ここが最大の効果。
- [x] **M-2f: 既定**。実 GPU があるとき `-b auto` / `-b cuda` は自動で新方式（`dispatch_update()` が MOTION_EST を device に置く）。CPU 単体は従来の hex 探索のまま（`-b cpu`）。
- [x] **ビット一致**: `AEG_SEED=1` で CPU の `me_batch` を通した場合とデバイスが **全 12 ピクチャでバイト一致**（1080p、B あり）。
- [x] **決定性**: フレームスレッド使用時、GPU 探索は参照ピクチャ全体の完成を待つ（`ref_wait`）ため、同じ入力·同じオプションで常に同じストリーム。CPU ワーカーの `seed_mv` もピクチャごとに初期化。
- **実測（1080p, p2, 12f, bframes 3）**:
  - 既定 CPU（フレームスレッド）1.03 s / GPU 1.75 s。CPU は 12 コアでスレッド並列、GPU はデバイス探索の待ち合わせで伸びる。
  - 1 スレッド同士: CPU 2.94–3.38 s / GPU 2.37–2.54 s（**1.2–1.4x**）。
  - ステージ内訳（-t1）: ME 1267 → 735 ms、transform 1591 → 1154 ms。
- 既知の不具合: 8x8 形状のサブペルラウンド（`AEG_S3SUB=1`）は 1080p で CPU/GPU のストリームがずれる。既定は 0。

### M-1 デバイス常駐＋非同期 — **完了（ピクチャ内）**
- [x] M-1a: `me_pic_upload` による luma ピクチャのデバイス常駐（slot 0 = 現フレーム、1 = 参照）。内容 ID でキャッシュし、同じピクチャの shape 0/3 と、同じ参照を使う B 群で再転送しない。
- [x] M-1b: レーンごとの `CUstream`（clone も自前の stream）、`cuMemcpyHtoDAsync` / `DtoHAsync`、**ピン留めステージング**（job / mv / sad を `cuMemAllocHost`）、OpenCL は `CL_FALSE` + `clFinish`。
- [x] M-1c: CUDA の全 `cuMemcpy*`/起動の戻り値検査、`nblk<=0` ガード、`AEG_DEVICE`/`--device`、**プライマリコンテキスト**、**一時的失敗のリトライ**（連続 3 回で無効化、失敗した呼び出しは CPU 参照で完結）。op 単位の無効化とデバイスロスト時の再初期化は未着手。
- [x] M-1d: `me_batch` は常駐ピクチャを使い、**1 呼び出し 1 同期**（CUDA は stream 同期、OpenCL は finish）。P は 2 shape、B は 2 list で 2 回。
- 実測（1080p, 12f, B あり, `-t1`）: ME 735 → **700 ms**。既定（フレームスレッド）: 1.72–1.82 s。
- **未実装**: 次ピクチャの ME を現ピクチャの CPU 処理中に先出しするクロスピクチャ先読み。現状のオーバーラップはフレームスレッドが担う。
- ゲート: 呼び出し ≤ 20 回/ピクチャ（達成: 2）、転送 ≤ 4 MB/ピクチャ（現フレーム 1 回＋新規参照のみ）。

### M-1.2 デバイス常駐再構成（luma+chroma、opt-in）
- [x] `AEG_REC_DEV=1`: deblock 終了後の参照ピクチャを device DPB slot（2..15）へ格納。CUDA は luma+chroma の 3 プレーン、OpenCL の CPU デバイス（PoCL）は luma のみにフォールバック。CPU 再構成は保持し、download は 0。
- 実測（1080p 12f, `AEG_SEED=1 -t1`）: CUDA **8 uploads / 30.16 MB / 0 MB download = 2.51 MB per picture**、0.67 calls/picture。OpenCL/PoCL は 19.00 MB / 1.58 MB per picture。
- `AEG_SEED=1 AEG_REC_DEV=1` で CPU/CUDA/OpenCL **ビット一致**、既定（opt-in なし）ともバイト一致。
- 既定 OFF。既定経路・`AEG_LA_BATCH=0`・`AEG_LA_DEV=1` は不変。デブロックの device 実行（M-4a）は未着手。OpenCL の chroma 常駐は `CL_DEVICE_TYPE_CPU` を検出して自動フォールバックし、警告を 1 行出力（PoCL 2.1 のクラッシュ回避）。実 OpenCL GPU はこの環境に存在しないため未検証。

### M-1.2d persistent intra 再構成（今回も凍結）
- `AEG_REC_GPU=1` の probe を追加: 実装せず、転送量を 1 行で報告する。1080p で full source+recon **5.98 MiB/pic**、luma-only **3.98 MiB/pic**、levels+modes 約 **6.23 MiB/pic**。M-1 の 4 MB ゲートは full で超過。
- persistent カーネルで呼び出しを 2/pic にしても、再構成に必要な **source upload + levels/modes upload + recon download** が 4 MB を超える。luma-only でも levels/modes を加えると超過。
- さらに現行 `code_mb()` はモード決定・量子化・再構成が密結合で、最終 MB の levels/modes を保持していない。device replay 用のコマンド記録を先に追加する必要があり、今回のタイムボックスでは閉じられない。
- 次回は機能系（psy-rd/CQM/weighted B/4:2:2/12bit/lossless/コンテナ）へ切り替える。M-1.2d は「source/recon を device 常駐させる M-1.2d-R」として再設計する。

### M-3 ルックアヘッド + mbtree の GPU 化（H-Q2）
- [x] **M-3.3a lowres リング常駐**
  - `aeg_be_ops.lowres_upload(slot, plane, stride, w, h, id)` と `lowres_sad_batch(cur_slot, ref_slot, ...)` を追加。
  - CUDA/OpenCL/CPU が `AEG_LOWRES_SLOTS`（96）個の lowres スロットを持ち、内容 ID が同じなら再転送しない。
  - encoder 側は `AEG_LA_DEV=1` で lookahead リングの低解像度プレーンを 1 ピクチャ 1 回アップロード（既定 off: 通常エンコードのストリーム・速度を変えない）。
  - `tests/dev_bench` に常駐 vs 毎回転送の比較を追加。1080p/64 slot/100 pair で **転送 118.75 → 38.00 MB（3.12x 減）**、device 時間 **70.2 → 28.6 ms（2.46x）**。
  - `AEG_LA_DEV=1` の 1080p 12f: 12 uploads / 7.12 MB、出力は既定と**バイト一致**。中央値 1.765 s（既定 1.731 s、+2% は upload とノイズ）。
- [x] **M-3.3b 全ペア一括 lowres カーネル**
  - `lowres_batch` カーネル：全ペア×全ブロックを 1 グリッドで処理。イテレーション（step 16/16/8/8/4/4/2/2/1/1）ごとにカーネルを切り替え、host は 10 launch + 1 sync。
  - ring を 1 個の連続バッファに変更（CUDA は slot オフセット、OpenCL は sub-buffer）。
  - CPU reference（`cpu_lowres_batch_search`）と CUDA/OpenCL が同一の候補順・ties・整数演算。`dev_bench` の 8 pairs × 8160 blocks で**結果一致**。
  - `AEG_LA_BATCH=1` で mbtree の計画を一括探索。1080p 12f: **4 calls / 4.39 MB / 0.33 calls per picture**（M-1 ゲート内）。
  - `dev_bench`: 8 pairs × 8160 blocks で CPU 372.67 ms / CUDA **6.56 ms**。
  - 既定 off。`AEG_LA_DEV=1` は従来どおり upload のみで出力は既定と**バイト一致**。
- [x] **M-3.1a/b 品質ゲート（PASS）**
  - 10 iteration（step 16/16/8/8/4/4/2/2/1/1）＋ batch 時 mbtree strength 1.5 に再チューニング。
  - 長尺 4 クリップ（240/300f）× CQP 4 点・CRF 4 点、PSNR/SSIM の BD-rate（AEG_LA_BATCH vs 既定）:
    - CQP: 平均 PSNR **-0.262%** / SSIM **-0.083%**（全クリップ ≤ +0.5%）
    - CRF: 平均 PSNR **-0.955%** / SSIM **-0.840%**（全クリップ負 = batch が少ないビット）
  - 旧既定（scalar lowres）は `AEG_LA_BATCH=0` で再現。`AEG_LA_DEV=1` は新既定と一致。
- [~] M-3.2 決定論的 mbtree（固定小数点）— **opt-in 凍結**。3 バックエンド一致は達成、品質ゲートは未達（Q16 でも CQP 平均 SSIM +2.788%）。詳細は本書末尾。
- [x] **M-3.4 品質・速度の最終再調整（既定 ON 決定）**
  - `AEG_LA_BATCH` を既定 ON（`AEG_LA_BATCH=0` で旧 scalar lowres）。batch 時 mbtree strength 1.5、旧経路 1.6。
  - 新既定 1080p/12f = **1.688 s**（197,721 バイト）。旧経路 = 1.661 s（202,923 バイト）。品質は新既定が有利で、速度差はノイズ範囲。
  - 品質（新既定 vs 旧経路、長尺4クリップ×CQP4/CRF4）:
    - CQP: 平均 PSNR **-0.262%** / SSIM **-0.083%**
    - CRF: 平均 PSNR **-0.955%** / SSIM **-0.840%**
  - 新既定での回帰: `AEG_SEED=1` CPU/CUDA バイト一致、open-GOP CPU/CUDA 一致、`AEG_LA_DEV=1` は新既定と一致。`AEG_LA_BATCH=0` は旧既定（202,923）とバイト一致。
  - 速度（3回中央値）: 新既定 1.688 s (1.665–1.705) / 旧経路 1.661 s (1.649–1.721) / ME 696.7 ms (641.5–722.9) / mbtree0 1.448 s (1.369–1.495) / mbtree1 1.729 s (1.695–1.735) / LA_DEV 1.718 s (1.658–1.767)。

### M-4 フィルタ・統計・カーネル工学
- [x] M-4a1: 内部エッジ分割の CPU 参照。`AEG_DEBLOCK_SPLIT=1` で MB ごとに「境界→内部」の4グループをラスタ順に実行し、既定とビット一致（`--slices 1/2`、B あり/なし、`AEG_SEED=1`、open-GOP、10bit）。
  - `AEG_DEBLOCK_SPLIT=2`（全 MB の境界→全 MB の内部の一括バッチ）は不一致（197,721 vs 197,665 バイト）。原因は境界エッジと内部エッジの相互参照で、H.264 8.7 のラスタ順を崩せないこと。
  - 設計: 内部エッジ k=1..3 は MB 外を読み書きせず、bS は q 自身の nz/t8/mv/qp だけで決まる。MB 内の隣接エッジは 3 サンプル近傍が重なるため k=1,2,3 の順序が必要。MB 間の内部エッジは独立なので 1 WI/MB のグリッドで 1 起動・1 同期にまとめられるが、境界エッジとのラスタ順インターリーブは別途保持しなければならない。
- [x] M-4a-R（CPU 参照）: `AEG_DEBLOCK_PERSIST=1` の persistent ラスタスケジューラ。MB ごとの依存カウンタで「左 (x-1,y)、上 (x,y-1)、左上 (x-1,y-1)、右上 (x+1,y-1)」を待ち、自身の垂直→水平を処理して解放する。`--slices 1/2`、B あり/なし、`AEG_SEED=1`、open-GOP、10bit で既定とビット一致。
  - 右上依存が必要な理由: MB(x+1,y-1) の左境界が MB(x,y-1) の右下を書き換え、MB(x,y) の上境界がそこを読む。左/上だけでは足りない。
  - 現在は単一スレッドのイベントループ（ラスタ走査で最小インデックスの ready を選ぶ）。同じカウンタを device の persistent カーネルがスピン待ちに使う設計。
- [ ] M-4a-R2（device 化の前提）: 現在の CPU 再構成パイプラインでは、deblock 入力の再構成 upload（luma+chroma 約 3.13 MB）+ MB メタデータ（約 0.56 MB）+ 出力 download（約 3.13 MB）= 約 6.8 MB/pic で M-1 の 4 MB ゲートを超える。M-1.2d の device 常駐再構成を入力にできれば upload 0、download 3.13 MB + メタデータ 0.56 MB = 3.7 MB/pic、起動 1 + 同期 1 = 2 calls/pic でゲート内。M-1.2d が凍結中のため、M-4a2 は「device 常駐再構成を入力にする」か、luma/chroma を分割する M-4a-R3（ストリップ分割）を先に決める。
- [ ] M-4a2: 上記 M-4a-R の CUDA/OpenCL カーネルとビット一致テスト（M-4a-R2 の転送ゲート解決後）。
- [x] M-4b（CPU SIMD）: PSNR per plane（`psnr_y/u/v`）と 8x8 窓 SSIM（`ssim_y`）の集計を SSE2 化。整数部分は int32 レーン→int64 の厳密累積、SSIM の浮動小数演算と窓順はスカラーと同一。`AEG_METRIC_SCALAR=1` の参照と 8bit/10bit で完全一致（全統計値がビット同一）。1080p 12f の PSNR ステージは **49.6 ms → 17.8 ms（約 2.8x、3回中央値）**。
  - device 集計は現行パイプラインでは source 3.13 MB + recon 3.13 MB の追加 upload が必要で、M-1 の 4 MB/pic ゲートを超える。recon が device 常駐（M-1.2b）でも source 3.13 MB + rec upload 2.51 MB + stats download で 5.6 MB 超。
- [ ] M-4b2（device 化の分割案）: M-1.2d で source と recon を同時に device 常駐させ、追加 upload を 0 にする。または luma-only SSIM/PSNR を 2.09 MB に絞っても往復 4.18 MB でゲート超過のため、変換（source upload）を再構成生成と同一カーネルに融合する必要がある。
- [ ] M-4c: ビット深度テンプレート（10bit で GPU 有効化）
- [ ] M-4d: `satd_batch` のサブブロック分割、`interp_plane` のインデックス最適化
- [x] M-4e（部分）: `--device n` / `AEG_DEVICE` で CUDA・OpenCL のデバイスを選択（列挙は先頭のみ）

### M-5 H.264 機能完成（H-F1〜H-F5）
- [x] M-5a（部分）: `--qpfile`（表示順 `frame qp`、-1 で自動）、`--zones s,e,qp`（繰り返し可、qpfile が優先）、`--roi x,y,w,h,off`（矩形内 MB の QP オフセット）。API は `frame_qp/n_frame_qp`、`zones/n_zones`、`rois/n_rois`。CQM は未着手
- [x] M-5b（部分）: `--intra-refresh n`（初回以外 IDR なし、n アンカーで全 MB を更新、recovery_point SEI）、`--slice-max-mbs n`、`--slice-max-size n`（上限超過で行数を半分にして再コード、最小 1 行）。
- [x] M-5c: open-GOP（`--open-gop`）。キーフレームは非 IDR I スライス、B は前 GOP を参照可、各 I に recovery_point SEI
- [x] M-5d（psy-rd 部分）: `--psy-rd <rd>,<trellis>` を追加。`psy_rd > 0` のとき、候補の輝度歪みを sqrt(|残差|) マスクで正規化して置換（`full_price()`、`psy_mb_sse()`）。既定 OFF はバイト一致。8bit/10bit、CABAC/CAVLC、FFmpeg 全数デコードを確認。psy-trellis は値のパースのみで未適用（M-5d2）。
  - マスク式を `AEG_PSY_MODE` で 3 案比較（0=マスク平均正規化、1=逆マスクでテクスチャを減衰、2=比率補正）。CRF 4点×長尺4クリップの最良は mode 1: 平均 PSNR **+0.20/+0.60/+2.52%**（0.5/1.0/2.0）、SSIM **+0.52/-1.68/-4.08%**。treeL は 0.5 で **-0.65%** だが、平均は正で採用基準を満たさない。CQP は mode 1 でさらに悪化（cockL +0.49、vtestL +0.25）。
  - 改善が出ない原因: psy-rd を最終 SSE ブレンドだけに適用し、SATD/intra モード決定/trellis/chroma に入れていない。マスクも量子化後残差で MB 単位。x264 の統合 psy-rd とは別物。→ opt-in 凍結。
  - 1080p 12f 速度: 既定 1.694s → psy 1.0 **2.150s（+27%）**、マスク前計算の最適化は未実施。
- [x] M-5e: `--aq-mode 0..3`（1 = variance、2 = auto-variance、3 = dark bias）。既存の `--aq <tenths>` は mode 1 のまま
- [x] M-5f（CQM 完成）: `--cqm flat|jvt|<file>`、SPS の `seq_scaling_matrix_present_flag` に 4x4 6 個 + 8x8 2 個、8x8 dequant 適用。**原因は SPS の 8x8 リストに PPS 専用の `useDefaultScalingMatrix8x8Flag` を書いていたこと**（SPS では不要、1 ビット余分で `log2_max_frame_num` 以降がずれる）。修正後、8bit CABAC/CAVLC・10bit で `--dump-recon` と FFmpeg デコードが一致。既定 flat は 197,721 バイト不変。
  - flat vs jvt CQP 4点×長尺4クリップ BD-rate（jvt が flat 比、正=悪化）: PSNR **+10.10/+6.79/+7.00/+4.51%**（平均 **+7.10%**）、SSIM **+1.38/+3.18/+1.46/+2.49%**（平均 **+2.13%**）。JVT 行列はこの実装では明確に悪化。
- [x] M-5g（部分）: HDR10 static metadata SEI（`--master-display`、`--max-cll`、payload 137/144）。weighted B、B16、ref16、Blu-ray 制約は未着手
- [x] M-5h（implicit weighted B / explicit 凍結）: `--weightb` で `weighted_bipred_idc=2`、POC 距離比の重み（denom 5、w0+w1=32）を bi-pred の luma/chroma に適用。8bit/10bit、CABAC/CAVLC、B-pyramid 0 で `--dump-recon` と FFmpeg デコードが一致。CPU/CUDA ビット一致。explicit は B 用 `pred_weight_table` を実装したが、スライスヘッダの PWT 長が合わず（deblocking_filter_idc 6）、`--weightb explicit` は明示エラーで無効化。既定 OFF は 197,721 バイト不変。BD-rate は未完了。
- [ ] M-5i: libavcodec wrapper、コンテナ連携

### M-7 移植性（Windows / MSVC / ARM）
- [ ] M-7a: `src/thread.h`（pthread/Win32）、`aeg_dl`（Windows ライブラリ名）
- [ ] M-7b: objcopy 依存の解消（二重コンパイル方式）
- [ ] M-7c: CMake、MSVC、ARM/NEON

### M-8 検証基盤
- [ ] M-8a: `dev_bench` / `--bench` の数値ベースラインと回帰ゲート
- [ ] M-8b: BD-rate 自動ゲート（H-Q1/H-Q2）
- [ ] M-8c: JM 参照デコーダ、HW デコーダ、コンテナ、ファジング

---

## 2. 実測ログ

環境: WSL2 Ubuntu 18.04 / gcc 7.5 / GeForce RTX 2060 SUPER (34 SM) / ffmpeg 3.4 / x264 CLI あり。
（`make OPENCL=1 CUDA=1 dev_bench` の値は `docs/DESIGN_ja.md` §2.1）

| 日付 | 計測 | 結果 |
|---|---|---|
| 11-08 | 1080p p2 `-b cpu` / `-b auto` | 2.01 s / 1.89 s（24f）バイト一致 |
| 11-08 | 640x352 p4 強制 `-s tq=cuda` | 63.5 s（CPU 2.96 s の 21 倍遅い） |
| 11-08 | `dev_bench 1920 1080` | sad 12.2x / satd 4.8x / tq16blk 0.008x |
| 11-08 | `make test`（ffmpeg 3.4 で LF 変換して実行） | 148 bit-exact / 40 はデコーダが 1 フレーム重複出力（値の不一致ゼロ） |
| 11-08 | 640x352 p2 P-only: 既定 hex / ラウンド CPU / ラウンド CUDA | 1.51 s / 3.07 s / 1.83 s（ME 502 / 2038 / 815 ms） |
| 11-08 | 640x352 **16x16 のみ**: 既定 / ラウンド CUDA | ME 119 ms / 657 ms（**GPU が 5.5 倍遅い**） |
| 11-08 | 1080p p2 P-only -t1: 既定 / ラウンド CPU / ラウンド CUDA | 6.50 s / 13.40 s / 7.09 s（ME 2690 / 9705 / 2749 ms） |
| 11-08 | 1080p p2 -t1 12f parts=15: 既定 / 新方式(cuda) | 2.99 s / 2.68 s（**1.11x**）、ME 1208 / 957 ms |
| 11-08 | 同上、ビット数 | 493,245 / 499,764（**+1.3 %**）、PSNR 49.37 / 49.35 |
| 11-08 | ホストループ最適化（除算除去）後 | ME 957 → 1043 ms（ノイズ範囲、**ストリーム不変**） |
| 11-08 | 同アルゴリズムの CPU 実装との比較 | ME 3363 ms（CPU）vs 957 ms（GPU）= **3.5x** |
| 11-08 | B 対応後 1080p p2 12f bframes3 `-t1` | CPU 2.94–3.38 s / GPU 2.37–2.54 s（**1.2–1.4x**） |
| 11-08 | 同 `-b cpu` / `-b cuda` 既定（フレームスレッドあり） | 1.03 s / 1.75 s、202,923 vs 204,071 バイト、PSNR 49.48 vs 49.45 dB |
| 11-08 | `AEG_SEED=1` CPU / CUDA（B あり 1080p 12f） | **全 12 ピクチャでバイト一致** |
| 11-08 | `backend_check`（1080 行を追加） | 348 チェック一致 |
| 11-08 | 既定 `-b cuda` を 3 回連続 | 同一 md5（**決定性確認**） |

### M-2a の結論（重要）

「全画面ラウンド＋MB 局所精密化」は**この形では目的の速度に届かない**。切り分けの結果:

1. カーネルは速い（sad_batch は 1080p で CPU の 12 倍）。
2. 同アルゴリズム同士なら GPU は CPU の **3.5 倍**（ME 957 vs 3363 ms）——デバイスは仕事をしている。
3. しかしラウンド方式は既定の per-MB hex より **探索仕事量が約 2.8 倍**ある（形状ごとに整数 5 ＋ サブペル 9 ラウンド）。
4. 往復回数（形状あたり 14 回）とホスト側の候補組み立てが加わり、正味 **1.11x** にとどまる。
5. 品質も +1.3 % ビット（同 PSNR）と劣る（ラウンドは隣接の決定済みベクトルをシードにできないため）。

→ **ラウンド方式を捨て、デバイス側で探索を完結させる（M-2b）**。1 スレッド = 1 ブロックで hex/UMH/サブペルを回し、1 ピクチャ 1 launch。ホストとの往復は候補と結果の 1 回ずつだけにし、アルゴリズムは CPU 実装と同一（＝ビット一致、品質劣化なし）。

---

## 3. 既知の環境上の注意

- このクローンは `core.autocrlf=true` のため C 以外のテキストが CRLF。`tests/run_tests.sh` は dash で直接動かない（`tr -d '\r'` してから `sh` で実行）。
- `repo/build/work/` に検証用クリップが残っている（gitignore 済み）。
- 削除系のコマンドはサンドボックスに拒否されることがある。一時ファイルは `build/work/` に作る。
