# 設計: x264/x265 同等以上を満たす GPU ハイブリッドエンコーダ

- 版: 1.0（2026-11-08）
- 対象 branch: `arena/589d1e45-acceleratedencode-gpu`
- 前提文書: `AUDIT_REPORT_ja.md`（機能・画質・移植性の監査）、`GPU_REVIEW_ja.md`（GPU 利用の再レビュー）。どちらもこのリポジトリの外（`Q:\Downloads\HybridEncode2\`）に置いてある。
- 本書の役割: その 2 つを「実行可能な設計と工程」に落とす。**実装済みの範囲は §10 に明記**し、残りは M-1〜M-8 のマイルストーンとして、受け入れ条件つきで定義する。

---

## 1. 「完成」の定義

v1.0 の合格条件は、次の 4 表がすべて緑になること。数値は x264/x265 を同一クリップ・同一条件で測った比較。

| # | 要求 | 合格条件（v1.0 ゲート） |
|---|---|---|
| G1 | 画質 | 14 クリップ + 1080p 実素材 4 本、CQP と CRF、PSNR/SSIM/VMAF で **BD-rate ≤ +1.0 %**（x264 veryslow 比）、**≤ +2.0 %**（x265 medium 比、同一コーデック比較のみ） |
| G2 | 機能 | `docs/PARITY.md` の x264/x265 機能表で「必須」欄が全緑（4:2:0/4:2:2/4:4:4、8/10/12 bit、lossless、CQM、zones/qpfile、open-GOP、ROI、intra-refresh、slice-max-size、weighted B、HRD、HDR メタデータ、コンテナ/FFmpeg 連携、HEVC Main/Main10） |
| G3 | 速度 | 1080p で preset 4 が x264 veryslow の **≥ 1.2 倍**、preset 2 が x264 medium の **≥ 1.0 倍**、preset 0 が x264 veryfast の **≥ 0.8 倍**。GPU 併用時は CPU 単独より **≥ 1.35 倍**（同一ビットストリーム、同一 CPU コア数） |
| G4 | 互換性 | FFmpeg / JM / HM 参照デコーダで全ケース再構成一致、HW デコーダ（NVDEC/QSV/VCN/VideoToolbox）で再生可、MP4/MKV/TS で mux/demux、ブラウザ（MSE/WebCodecs）で再生、Blu-ray 制約モード |

G3 の「GPU 併用時 ≥1.35 倍」が本プロジェクトの存在意義であり、現状は **0.4 倍**（実測、§2）。ここが最初に直すべき数値。

---

## 2. 実測: 現在のデバイスは何をしているか

環境: Windows 11 + WSL2 Ubuntu 18.04、**GeForce RTX 2060 SUPER (34 SM)**、ドライバ 591.74、NVRTC 12.1、gcc 7.5。すべて本書のための実測。

### 2.1 カーネル単体（`make dev_bench` → `./build/dev_bench 1920 1080`）

1080p = 8,160 MB、1 ラウンド 73,440 候補。

| 操作 | CPU | device | CPU/device |
|---|---:|---:|---:|
| `sad_batch`（1 ラウンド全体） | 12.29 ms | **1.09 ms** | **11.3x** |
| `satd_batch`（1 ラウンド全体） | 4.65 ms | **0.92 ms** | **5.1x** |
| `ref_upload`（参照 4 プレーン） | (no-op) | 1.05 ms | — |
| `tq_batch` **16 ブロック**（現行の呼び方） | 0.40 **µs** | **98 µs** | **0.004x** |
| `tq_batch` 1 ピクチャ全体（130,560 ブロック） | 3.84 ms | 1.07 ms | 3.6x |

読み取れること:

1. **カーネルは十分速い。** 全画面 SAD は CPU の 11 倍。問題はカーネルではない。
2. **呼び出しの固定費が ~90〜100 µs ある。** 16 ブロックの係数バッチが 98 µs かかる一方、130,560 ブロックでも 1.07 ms。つまり中身は 0 に近く、すべて往復のコスト。**MB 単位で送る限り、1 フレーム 8,000 回 × 98 µs = 780 ms が確定で乗る**（1080p）。
3. **係数ステージは「1 ピクチャ一括」なら 3.6 倍速い。** ただし後述のとおり、この形は現在の逐次依存では実現できない（§4.3）。
4. `ref_upload` が 1.05 ms/ピクチャ。ラウンドごとに同じフレームを送り直す現設計では、1080p 1 枚あたり 40〜50 MB を無駄に転送している。

### 2.2 エンコード全体

| 構成 | 結果 |
|---|---|
| 640x352, p4, 60f: `-b cpu` | 2.96 s（20.3 fps） |
| 同上 `-b auto`（**修正前**） | **8.69 s（6.9 fps）= 2.9 倍遅い**。係数ステージが device に置かれていた |
| 同上 `-b auto`（**修正後**） | 2.96 s、`-b cpu` とバイト一致 |
| 同上 `-c cuda -s tq=cuda -s iq=cuda`（強制） | **63.5 s = 21 倍遅い**（ストリームは一致） |
| 1920x1080, p2, 24f, `-b auto` | 1.89 s、`-b cpu` とバイト一致 |
| 1920x1080, p2, `AEG_SEED=1 AEG_PARFULL=1`（全画面ラウンド ON） | ME ステージ 873 ms。**同じ探索を CPU で 1457 ms**（1.7 倍）だが、**既定の六角形探索は 852 ms** なので、GPU 版は「遅い探索を 1.7 倍速くした」だけで**正味の高速化ゼロ** |

`AEG_SEED=1 AEG_PARFULL=1` の ME 内訳（`AEG_ME_TIME=1`）: `me_seed_frame` 全体で 16〜31 ms/枚、うち整数ラウンドは 1.3〜6 ms。残りはサブペルのラウンド。**GPU ラウンドは 1 枚あたり数 ms しかないのに、`analyse_mb` の MB 単位探索が 70 ms/枚かかっている**（partition 形状ごとに 4〜5 回）。GPU ラウンドはそのうち 16x16 の 1 本しか置き換えていない → **足し算になっている**。

### 2.3 正しさ

- `backend_check`: 実 GPU 上で **312 オペレーションすべて CPU 参照と一致**。
- `-b auto` / 強制 device / `-b cpu` の出力はすべてバイト一致（1080p、640x352 で確認）。
- したがって**ビット一致の枠組みは実機で成立している**。壊れているのは配置と構造だけ。これは良い知らせで、以下の設計はその枠組みを保ったまま進められる。

---

## 3. 設計原則

**P1. デバイスに送るのは「1 発で 10^5 個以上の独立仕事があり、往復が 100 µs を超えない」ステージだけ。**
実測の固定費 90〜100 µs が判断基準。逆に言えば、1 ピクチャあたりの呼び出し回数の上限は ~20 回。現在は 8,000 回。

**P2. ピクチャは 1 回だけデバイスに載せ、DPB ごと常駐させる。**
現フレームも参照フレームも、ラウンドごとに再送しない。`aeg_dev_pic` ハンドルで管理し、ラウンド間の転送をゼロにする。

**P3. アルゴリズムは 1 つ、実装は 2 つ。**
CPU 実装と GPU 実装は同じ決定を出さなければならない（現在の係数ステージはこれで守られている）。**動き探索だけがこの原則を破っている**（全画面ラウンド vs 六角形歩行）ため、配置を時間で決められず、GPU が永久に選ばれない。探索を 1 本化して初めて「配置は純粋な速度ノブ」になる。

**P4. 配置は測定で決めるが、決めるのは「速い方」ではなく「速いことが証明できた方」。**
現在の `gpu_ns > 0 && cpu_ns > 0` はデッドロックし、`coef_pick` はノイズで誤判定した（2.9 倍遅い方を選んだ）。プローブは 1 段だけ、明示的に、コストモデル（固定費 × 呼び出し回数 + 実測スループット）で行う。

**P5. 1 回の失敗でデバイスを殺さない。**
op 単位の無効化、N 回連続失敗でその op のみ恒久 CPU、VRAM 逼迫時はバッファ縮小、デバイスロスト時は再初期化。現在は 1 回の失敗で全レーン全ステージが恒久 CPU。

**P6. 数値は CI で守る。**
正しさは既存の conformance、性能は `dev_bench` と `--bench` の数値をベースラインとして記録し、回帰を検出する（§8）。

---

## 4. 目標アーキテクチャ

### 4.1 ステージ配置（最終形）

実測に基づく最終配置。**「捨てる」と書いたものは二度とデバイスに置かない**（現在の設計の最大の誤りは、ここを測定で決めようとしたこと）。

| ステージ | 配置 | 根拠 |
|---|---|---|
| 動き探索（整数ラウンド、全 partition 形状） | **device** | 全画面バッチで 11 倍。呼び出しは 1 ピクチャ 4〜8 回 |
| 動き探索（サブペル SATD ラウンド） | **device** | 5 倍、呼び出し 6〜10 回 |
| 参照・現フレームのアップロード | device（1 回/ピクチャ） | 転送は 1 ピクチャ 3〜4 MB のみ |
| サブペル平面生成（`interp_plane`） | **device**（常駐参照から） | 全サンプル独立、1 ピクチャ 3〜6 launch |
| ルックアヘッド（低解像度探索・mbtree 伝播） | **device**（新規） | 現状 100 % CPU。ME と同じ形で、画質にも効く |
| デブロック | **device**（新規、4 フェーズ波面） | 全 MB 独立ではなく「隣接の再構成済みサンプルを読む」だけが制約。フェーズ分割で並列化可 |
| PSNR/SSIM | **device**（新規） | 純粋なリダクション |
| 4x4 変換・量子化・逆変換 | **CPU 固定** | 1 ピクチャ一括なら 3.6 倍だが、逐次依存で一括化できない。MB 単位では 245 倍の損。**デバイスに置かない**（`-s` でも警告を出す） |
| エントロピー（CABAC/CAVLC） | CPU 固定 | 本質的に直列 |
| イントラ予測・再構成 | CPU 固定 | 隣接依存 |
| モード判定（RD） | CPU 固定 | 直前 MB の決定に依存 |

この表の含意: **GPU の担当は「探索」と「フィルタ」と「統計」であり、変換でも量子化でもない。** 現在の実装はその逆をやっている。

### 4.2 デバイス常駐ピクチャと非同期パイプライン（M-1）

```c
/* 追加する内部 API（src/backend.h） */
typedef struct aeg_dev_pic aeg_dev_pic;   /* デバイス上の 1 ピクチャ */

/* 確保・解放。スロットは DPB の数だけプールする */
int  aeg_dev_pic_new(aeg_be_ops *o, int w, int h, aeg_dev_pic **out);
void aeg_dev_pic_free(aeg_be_ops *o, aeg_dev_pic *p);

/* ホスト → デバイス（非同期。完了はフェンスで待つ） */
int  aeg_dev_upload(aeg_be_ops *o, aeg_dev_pic *p, const pixel *const plane[3],
                    const int stride[3]);
/* デバイス → ホスト（探索結果だけ。数百 KB） */
int  aeg_dev_download(aeg_be_ops *o, aeg_dev_pic *p, int which, void *dst, size_t n);

/* 1 ピクチャ分の探索をまとめて投入する（内部で 10〜20 launch、1 回だけ待つ） */
int  aeg_dev_me_rounds(aeg_be_ops *o, const aeg_dev_pic *src,
                       const aeg_dev_pic *ref[2], const aeg_me_req *req,
                       aeg_me_result *res);
```

- **ストリーム**: レーンごとに 1 本の `CUstream` / out-of-order queue。レガシー NULL ストリームは使わない（暗黙同期でオーバーラップが消える）。CUDA は `cuMemcpyHtoDAsync_v2` + `cuStreamSynchronize`、OpenCL は `clEnqueueWriteBuffer(...CL_FALSE)` + event。
- **ピン留め**: ホスト側のステージングバッファは `cuMemAllocHost` / `CL_MEM_ALLOC_HOST_PTR` で確保。
- **パイプライン**: `aeg_encoder_encode()` の内部を「N 枚目のエントロピーを CPU が書いている間に、N+1 枚目の探索を GPU が回す」形にする。フレームスレッド機構（`ft_*`）は既にあるので、そこに「デバイス作業キュー」を 1 本足す。これが G3（≥1.35 倍）の主要因になる。
- **VRAM**: 1080p で 1 ピクチャ ≈ 3 MB（4:2:0 8bit + パディング）。DPB 8 + 現フレーム 1 + 作業 2 = 11 枚で 35 MB。4K でも 140 MB。上限を超える場合は解像度ではなく DPB を減らす。

### 4.3 探索の一本化（M-2）

現在 2 つある探索を 1 つにする。

```
入力: 現フレーム（device 常駐）、参照フレーム（device 常駐）、探索窓
1) シード生成（低解像度 lookahead のベクトル、前フレームの同位置、零）
2) 整数ラウンド（step 8/4/2/1、9 近傍、全 MB 同時）        ← 1 launch/round
3) サブペルラウンド（half → quarter、必要な MB だけ）        ← 1 launch/round
4) MB 局所精密化：得られたベクトルの周囲 ±1 サンプルだけを CPU で評価
   （MV 予測子との差分コストを正しく入れるため。ここは MB 数 × 数点で軽い）
5) partition 形状（16x8/8x16/8x8）も 2)〜4) と同じバッチに載せる
   （形状ごとに候補配列を追加するだけで、カーネルは同じ）
```

これで **CPU 実装と GPU 実装が同一の決定を出す**。`aeg_me_block_t()` の六角形歩行は「preset 0 の高速モード」として残すが、既定は上のパイプラインにする。

画質の検証が必須（G1）。現状の全画面ラウンドは「16x16 のみ・予測子コスト無し」なので品質が落ちる可能性がある。**M-2 の合格条件は BD-rate ≤ +0.5 %**。もし超えるなら、4) の局所精密化の半径を広げる（±2、±3）か、ラウンドの step を細かくする。速度とのトレードオフは GPU 側で吸収できる（11 倍の余裕がある）。

### 4.4 描画ポリシー（M-0 で実装済み）

```
配置決定 = 
  強制 (-s stage=be)                 → そのとおり
  CPU 専用ステージ（§4.1 の表）        → CPU
  device が無い / CPU デバイス(PoCL)   → CPU（-b opencl を明示した時のみ device）
  それ以外                            → コストモデル:
        T_dev = calls * fixed_us + work / dev_throughput
        T_cpu = work / cpu_throughput
        T_dev * 1.15 < T_cpu のときだけ device
  fixed_us / throughput は起動時に 1 回だけ実測（aeg_backend_calibrate を
  ディスパッチャに接続し、aeg_device_info にも入れる）
```

現在の「両方の EMA が > 0 になったら比較」は、device で走らないと EMA が生まれないデッドロックなので廃止する。

### 4.5 カーネル設計指針（M-4）

- **候補配列は candidate-major**（`idx = cand * nblk + blk`）。warp 内の隣接レーンが隣接 MB を読むので、`sad_batch` のロードがコアレスになる。※実測では 1080p でほぼ差が無かった（L1 が効いている）が、4K と ADP の SATD では効くので統一する。
- ただし **`satd_batch` は現状 1 ワークアイテム = 1 候補で 16×16 を直列に舐めており、`bw×bh` が 8x8 のとき 4 倍無駄がある**。ワークアイテムを「候補 × 4x4 サブブロック」に分割し、リダクションを warp 内で行う。
- **共有メモリ**: 参照タイル（64×64 + 2 サンプルのハロー）をブロック共有メモリに載せ、9 近傍の候補で再利用する。
- **8x8 変換カーネルが無い**。High profile + trellis の主力なので、M-4 で追加する（ただし §4.1 のとおり係数ステージは CPU 固定なので、これは「将来 1 ピクチャ一括が可能になった場合」の備えであり、優先度は低い）。
- **ビット深度**: カーネルを `pixel` 型でテンプレート化（`uchar` / `ushort`）。現在は 8bit 専用で、10bit は GPU 完全無効。
- **`interp_plane` の `i % ow` / `i / ow`** を 2D NDRange か事前計算に置き換える。

### 4.6 エラーとデバイス管理（M-1）

- CUDA の `cuMemcpy*` / `cuLaunchKernel` の戻り値を全部見る（現在は 12 箇所で無視、静かなデータ破壊）。
- `cuCtxCreate` → `cuDevicePrimaryCtxRetain`（ホストアプリの current context を奪わない、終了時に pop 不要）。
- デバイス選択: 全プラットフォーム・全デバイスを列挙し、`is_gpu` で絞り、`--device N` と `AEG_DEVICE` で選べるようにする。iGPU + dGPU の同時利用（レーンを分ける）は M-4 以降。
- 失敗時の扱い: op 単位のカウンタ（3 回連続失敗でその op のみ無効化）、VRAM 不足はバッファ縮小リトライ、`CUDA_ERROR_LAUNCH_TIMEOUT` / `DEVICE_LOST` はコンテキスト再作成。

---

## 5. 機能パリティ（G2）と作業量

`docs/PARITY.md` に全項目を列挙する。優先度は「x264/x265 の実利用で必須か」で決める。

### 5.1 H.264（既存コーデックの完成）

| 項目 | 現状 | 優先 | 規模 |
|---|---|---|---|
| 4:2:2 / 4:4:4 出力 | 無 | 高 | M（入力変換 + MB 層 + 変換/量子化のクロマ分岐） |
| 10bit（CPU） | 有 | — | — |
| 10bit（GPU） | 無 | 高 | S（カーネルの型テンプレート化） |
| 12bit | 無 | 中 | M |
| lossless（`--qp 0`） | 無 | 中 | M（変換スキップ + PCM） |
| カスタム量子化行列 | 無 | 中 | M（SPS/PPS + スケーリング） |
| zones / qpfile | 無 | **高** | S（API + RC フック） |
| open-GOP | 無 | 高 | M（CRA/BLA 相当 + recovery point SEI） |
| ROI | 無 | 中 | S（AQ の一種として実装） |
| intra-refresh / slice-max-size / slice-max-mbs | 無 | **高**（配信で必須） | M |
| weighted B | 無 | 中 | S |
| B 最大 16 / ref 最大 16 | 3 / 8 | 中 | S |
| psy-rd / psy-trellis | 無 | **高**（画質の体感差の主因） | M |
| AQ モード（auto/variance/dark） | variance のみ・既定 off | 高 | S |
| HDR10/HLG mastering display / CLL | 無 | 高 | S（SEI のみ） |
| MBAFF / PAFF | 有 | — | — |
| MVC / SVC / Blu-ray 制約 | 無 | 低 | L |
| libavcodec wrapper / FFmpeg filter | 無 | **高**（実利用の入口） | M |
| コンテナ（MP4/MKV/TS）連携 | 生 Annex-B のみ | 高 | S（dts/pts は既にある） |

### 5.2 HEVC（x265 と並ぶための必須項目）

M-6。H.264 の上に積むのではなく、**共有部（バックエンド、変換、ビットライタ、レート制御、テスト基盤）を流用しつつ、CTU/CU ツリーと CABAC コンテキストを新規に書く**。規模は H.264 実装の 1.5〜2 倍。段階:

1. HEVC Main（8bit 4:2:0）: VPS/SPS/PPS、CTU 64、QUAD/RECT 分割、35 intra モード、merge/skip、AMVP、8 タップ/4 タップ補間、SAO、DBF、CABAC（HEVC）、WPP + tiles。
2. Main10。
3. 4:2:2 / 4:4:4 / 12bit、lossless、CQM、ALF（x265 の `--alf` 相当は任意）。

GPU 側は M-1〜M-4 の探索基盤がそのまま使える（CTU 単位の SAD/RD 候補評価はむしろ GPU 向き）。

---

## 6. マイルストーン

各マイルストーンは独立にマージ可能で、受け入れ条件を満たさない限り次に進まない。

### M-0 配置と正しさの土台（**実装済み**、§10）
- デバイス CPU 判定（PoCL を自動配置から除外）、`-s stage=auto` の意味修正、係数ステージの自動配置撤去、自動配置デッドロックの解消、GCC 7 ビルド、`--bench` の kb/s 修正、`dev_bench` 追加。
- 受け入れ: `-b auto` が `-b cpu` とバイト一致かつ遅くない（実測 1080p 1.89 s vs 2.01 s）。`backend_check` 312 一致。

### M-1 デバイス常駐 + 非同期（G3 の土台）
- `aeg_dev_pic`、レーンごとのストリーム、ピン留めステージング、`aeg_dev_me_rounds()`（1 ピクチャ 1 同期）、CUDA の戻り値検査とプライマリコンテキスト、op 単位の失敗処理。
- 受け入れ: 1080p の探索で **呼び出し 20 回/ピクチャ以下**、転送 ≤ 4 MB/ピクチャ（現状 40〜50 MB）、`dev_bench` の `sad_batch` が現状比 ≥ 1.2 倍、全ストリーム一致。

### M-2 探索の一本化
- §4.3 のパイプラインを唯一の探索に。16x8/8x16/8x8 もバッチ化。六角形歩行は preset 0 のみ。
- 受け入れ: **BD-rate ≤ +0.5 %**（14 クリップ、CQP と CRF）、GPU 併用で 1080p preset 2 が CPU 単独比 **≥ 1.4 倍**、`-b cpu` と `-b cuda` がバイト一致。

### M-3 ルックアヘッドと mbtree の GPU 化
- 低解像度探索と mbtree 伝播をバッチ化。CRF 長尺の +11.8 % の主因に効く。
- 受け入れ: CRF 長尺 BD-rate（x264 veryslow 比） ≤ +5 %、速度は 1080p で ≥ 1.1 倍。

### M-4 フィルタとカーネル工学
- デブロックの 4 フェーズ波面カーネル、PSNR/SSIM、`satd_batch` のサブブロック分割、共有メモリのタイル、ビット深度テンプレート、全プラットフォーム列挙と `--device`。
- 受け入れ: デブロックが全解像度でストリーム一致、1080p で deblock ステージ ≥ 2 倍、10bit で GPU が有効。

### M-5 H.264 機能完成（G2 の H.264 部分）
- §5.1 の「高」項目すべて。libavcodec wrapper とコンテナ連携を含む。
- 受け入れ: `docs/PARITY.md` の H.264 必須欄が全緑、FFmpeg から `-c:v aeg` で使える。

### M-6 HEVC（G2 の HEVC 部分）
- §5.2。Main → Main10 → 高 bit depth。
- 受け入れ: HM 参照デコーダ一致、x265 medium 比 BD-rate ≤ +2 %（同一コーデック）、1080p preset 相当で ≥ 0.8 倍速。

### M-7 移植性とパッケージング
- CMake、Windows（スレッド抽象化、`LoadLibrary`、`objcopy` 依存の解消＝10bit 多重化をプリプロセッサで行う方式へ）、MSVC、macOS、ARM/NEON、pkg-config/vcpkg/Homebrew。
- 受け入れ: 5 プラットフォーム × 3 コンパイラの CI が緑。

### M-8 検証基盤
- 実 GPU を持つ CI ランナー（1 台でよい）、`dev_bench`/`--bench` の数値ベースラインと回帰ゲート、BD-rate/VMAF の自動ゲート、HW デコーダ・ブラウザ・コンテナの適合試験、ファジング（入力 YUV、API、ビットストリーム、デバイスバックエンド）。
- 受け入れ: すべての PR で G1/G3 の数値が自動比較される。

---

## 7. 検証戦略

| 層 | 手段 | 頻度 |
|---|---|---|
| カーネル正しさ | `tests/kernel_check.c`（CPU 上でカーネル文字列を実行）、`tests/backend_check.c`（実 GPU で 312 ケース） | 毎コミット |
| ストリーム一致 | `make test`（78 ケース、bit-exact）、`tools/gpu_validate.sh`（CPU/auto/強制 device のバイト一致） | 毎コミット |
| デコーダ一致 | FFmpeg でデコードし再構成と比較（既存）、JM/HM 参照デコーダ | 週次 |
| 画質 | `tools/bdrate.py`（既存）+ VMAF/SSIMULACRA2、14 クリップ + 1080p 実素材 | マイルストーン毎 |
| 性能 | `dev_bench`（カーネル単体）、`--bench`（エンコード全体）、ステージ内訳 `-v 2` | 毎コミット（数値ベースライン比較） |
| 互換性 | HW デコーダ 4 種、MP4/MKV/TS、ブラウザ、Blu-ray 制約 | リリース前 |

**現状の弱点**: 性能の自動比較が存在しない。M-0 で `dev_bench` を入れたのが最初の一歩で、M-8 で CI に載せる。

---

## 8. 移植性・ライセンス

- **Windows/POSIX の抽象化**（M-7 の前提）: `pthread` → `src/thread.h`（`pthread` / Win32 / C11 threads）、`dlopen` → `aeg_dl`、`objcopy --redefine-syms` による 10bit 多重化 → `#include` による二重コンパイル（x264 と同じ方式。ビルド時間は増えるが binutils 依存が消え、LTO とデバッグ情報が壊れない）。
- **ライブラリ名**: `OpenCL.dll` / `nvcuda.dll` / `nvrtc64_*.dll` を追加（現状 Linux 名のみ）。
- **ライセンス**: 現状 MIT だが、x264 由来のアルゴリズム名・概念が多数。**来歴を `NOTICE` に記録**し、コードのコピーが無いことを確認する。x264 から移植する場合は GPLv2+ になるため、MIT 配布を続けるなら独立実装を維持する。H.264/H.265 の特許プールについては `PATENTS.md` で明示（MIT は特許実施権を付与しない）。

---

## 9. リスク

| リスク | 影響 | 対応 |
|---|---|---|
| M-2 の一本化で画質が落ちる | G1 未達 | 局所精密化の半径を広げる。GPU には 11 倍の余裕がある（§4.3） |
| HEVC の規模（M-6） | 全体スケジュール | 共有部を先に固める。HEVC は独立ブランチで並行開発できる構造にする |
| 固定費 90〜100 µs（WSL 固有の可能性） | GPU 併用の上限 | まずベアメタル Windows/Linux で測り直す（M-1 の最初の作業） |
| ライセンス（GPL 混入） | 再配布不能 | 来歴管理とレビューゲート |
| 特許（H.264/H.265 プール） | 商用利用時の費用 | `PATENTS.md` で明示。要件次第で VP9/AV1 等の代替も検討 |

---

## 10. 今回実装したこと（M-0）

| 変更 | ファイル | 内容 |
|---|---|---|
| GCC 7 / clang < 12 対応 | `src/mc.c` | `_mm_storeu_si32` の移植性シム（upstream は GCC 11 でしかビルドできなかった） |
| デバイスの CPU 判定 | `src/backend.{c,h}`, `src/backend_opencl.c` | `aeg_be_ops.is_gpu`、`aeg_dispatcher.auto_ok`。PoCL は自動配置の対象外（`-s` では使える） |
| `-s stage=auto` の意味 | `src/backend.c` | `AEG_BACKEND_AUTO` を「CPU 強制」ではなく「auto」に |
| 係数ステージの自動配置を撤去 | `src/backend.c` | 実測 2.9〜21 倍遅いため既定 CPU。`AEG_COEF_PROBE=1` で旧試行を再現可能 |
| 自動配置デッドロック | `src/backend.c` | `auto_ok` で明示的に判断、`-s` 以外で device に置かないステージを明示 |
| OpenCL の細部 | `src/backend_opencl.c` | `sse_batch` の `count==0` で `*out=0`、`tq`/`iq` の `nblk<=0` ガード |
| `--bench` の bitrate | `cli/aegenc.c` | 単位バグ（速度が遅いほど数字が増える式）を修正 |
| ME の所要時間デバッグ | `src/encoder.c` | `AEG_ME_TIME=1` で 1 枚あたりの探索時間を出力 |
| デバイス性能ハーネス | `tests/dev_bench.c`, `Makefile` | `make dev_bench`。§2.1 の数値はこれで取った |

**未着手（次の作業）**: §4.2 の常駐・非同期基盤（M-1）から。ここが G3 の前提であり、`dev_bench` の数値がそのまま設計目標になる。

---

## 11. 追加実装（M-2 / M-4 / M-5 の今回分）

| 変更 | ファイル | 内容 |
|---|---|---|
| 全画面整数探索 `me_batch` | `src/backend.{c,h}`, `kernels.h`, `backend_opencl.c`, `backend_cuda.c` | 1 ブロック = 1 ワークアイテム。CPU 参照と CUDA/OpenCL カーネルが同一の nine-point descent を行い、ベクトルと SAD を返す。 |
| B ピクチャの全画面探索 | `src/encoder.c` | `me_seed_frame_b()` が list 0/1 をそれぞれ探索。`analyse_mb_b()` は `aeg_me_refine_around()` で MV 予測子込みの局所精密化だけを行う。 |
| 既定で device 探索 | `src/backend.c` | 実 GPU があるとき MOTION_EST を自動で device に置く。PoCL は `auto_ok` で除外したまま。 |
| 決定性 | `src/encoder.c` | フレームスレッドでは参照全体を待ってから全画面探索する。ワーカーの `seed_mv` をピクチャごとに初期化。同じ入力・同じオプションで同一ストリームを確認。 |
| 画質 | `src/encoder.c` | `aeg_me_refine_around()` の結果を使い、1080p B ありで `-b cpu` 比 -0.56% バイト（同 PSNR）。 |
| 統計 API | `src/encoder.c` | `psnr_y/u/v` と 8x8 窓 `ssim_y` を実装。PSNR ステージの時間を計測。 |
| デバイス選択 | `backend_opencl.c`, `backend_cuda.c`, `cli/aegenc.c` | `--device n` / `AEG_DEVICE`。範囲外は警告して 0 にフォールバック。 |
| CUDA 堅牢性 | `src/backend_cuda.c` | 全 `cuMemcpy*` の戻り値検査、`nblk<=0` ガード、SSE のアップロード検査。 |
| `--qpfile` | `include/aeg/aeg.h`, `cli/aegenc.c`, `src/encoder.c` | `frame qp` 行（-1 = 自動）。API は `aeg_params.frame_qp/n_frame_qp`。QP は RC・picture offset の後に上書きされ、AQ/trellis はそこから動く。 |
| テスト | `tests/backend_check.c` | 1080 行（16 の倍数でない高さ）を追加、348 チェック。 |

**M-1（ピクチャ内）完了**: `me_pic_upload` による luma 常駐、ID キャッシュ、レーンごとの CUDA stream、非同期コピー、ピン留め staging、OpenCL の `CL_FALSE`+finish、1 呼び出し 1 同期。クロスピクチャ先読みは未実装（フレームスレッドがオーバーラップを担う）。

**M-3.3a 完了**: lowres リングの device slot（96）と `lowres_upload` / `lowres_sad_batch`、内容 ID キャッシュ。`AEG_LA_DEV=1` で encoder から 1 ピクチャ 1 upload。dev_bench 1080p/64 slot/100 pair で転送 118.75 → 38.00 MB、device 70.2 → 28.6 ms。

**M-3.3b 完了**: `lowres_batch` カーネル（全ペア×全ブロック、step ごとに 1 launch、host は 1 sync）、連続 ring、CPU/GPU 同一結果。`AEG_LA_BATCH=1` で 1080p 12f は 4 calls / 4.39 MB / 0.33 calls per picture。dev_bench 8 pairs で CPU 372.67 ms / CUDA 6.56 ms。

**M-3.1a/b 品質ゲート PASS / M-3.4 で既定 ON**: 10 iteration + batch 時 mbtree strength 1.5。長尺 4 クリップ × CQP/CRF 4 点で CQP 平均 PSNR -0.262% / SSIM -0.083%、CRF 平均 PSNR -0.955% / SSIM -0.840%（新既定 vs 旧 scalar 経路、負 = batch が少ないビット）。1080p/12f は新既定 1.688 s（197,721 バイト）、旧経路 1.661 s（202,923 バイト）。`AEG_LA_BATCH=0` で旧経路。

**未着手**: M-1 のクロスピクチャ先読み、M-3.2（決定論的 mbtree）、M-4a（デブロック波面）、M-4c（10bit device）、M-5 の残り（psy/CQM/weighted B/4:2:2/4:4:4/12bit/lossless/コンテナ）、M-6（HEVC）、M-7（移植性）、M-8（実機 CI・ファジング）。
