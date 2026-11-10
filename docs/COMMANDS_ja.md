# 検証済みコマンド集（日本語）

作業ディレクトリはリポジトリ直下。WSL の例:

```sh
cd /mnt/q/Downloads/HybridEncode2/repo
```

`build/aegenc` は `make -j12 OPENCL=1 CUDA=1` で作成する。
入力は `build/work/1080p.yuv`（1920x1080, 12 frames）と
`build/work/small.yuv`（320x192, 4 frames）を想定。

## 1. 既定エンコード

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 -b cuda -o build/work/default.264 -v 0
stat -c%s build/work/default.264
```

期待値: **197721**。

旧 scalar lowres 経路:

```sh
AEG_LA_BATCH=0 ./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 --weightb 0 -b cuda \
  -o build/work/old.264 -v 0
stat -c%s build/work/old.264
```

期待値: **202923**。
旧既定そのもの（`AEG_LA_BATCH=1 --weightb 0`）は **197721**。

## 2. CPU/CUDA バイト一致

```sh
AEG_SEED=1 ./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 -b cuda -o build/work/seed_cuda.264 -v 0
AEG_SEED=1 ./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 -b cpu -o build/work/seed_cpu.264 -v 0
cmp build/work/seed_cuda.264 build/work/seed_cpu.264 && echo SAME
```

期待値: `SAME`。

## 3. open-GOP

```sh
AEG_SEED=1 ./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 --open-gop -b cuda \
  -o build/work/og_cuda.264 -v 0
AEG_SEED=1 ./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 --open-gop -b cpu \
  -o build/work/og_cpu.264 -v 0
cmp build/work/og_cuda.264 build/work/og_cpu.264 && echo SAME
```

## 4. weighted B

implicit（既定）:

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 --b-pyramid 0 --weightb implicit \
  -b cuda -o build/work/wb_implicit.264 -v 0
```

explicit:

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 --b-pyramid 0 --weightb explicit \
  -b cuda -o build/work/wb_explicit.264 -v 0
```

`--weightb 0` で無効化。

## 5. CQM

flat（既定）:

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 4 \
  -p 2 -q 27 --bframes 0 -b cuda -o build/work/cqm_flat.264 -v 0
```

jvt + 再構成一致:

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 4 \
  -p 2 -q 27 --bframes 0 -b cuda --cqm jvt \
  --dump-recon build/work/cqm_rec.yuv -o build/work/cqm_jvt.264 -v 0
ffmpeg -v error -y -i build/work/cqm_jvt.264 \
  -f rawvideo -pix_fmt yuv420p build/work/cqm_dec.yuv
cmp build/work/cqm_rec.yuv build/work/cqm_dec.yuv && echo SAME
```

CQM の 8x8 のみが変わり、既定 flat は scaling list を送出しない。

## 6. コンテナ出力

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 -b cuda --mux mp4 build/work/out.mp4 -v 0
ffprobe -v error -show_entries stream=width,height,avg_frame_rate,nb_frames \
  -of default=nw=1 build/work/out.mp4
ffmpeg -v error -i build/work/out.mp4 -f null - && echo DEC_OK
```

MKV:

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 -b cuda --mux mkv build/work/out.mkv -v 0
ffmpeg -v error -i build/work/out.mkv -f null - && echo DEC_OK
```

TS:

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 -b cuda --mux ts build/work/out.ts -v 0
ffprobe -v error -show_entries stream=width,height,avg_frame_rate \
  -of default=nw=1 build/work/out.ts
ffmpeg -v error -i build/work/out.ts -f null - && echo DEC_OK
```

TS は `avg_frame_rate=30/1`、`r_frame_rate` は 60/1 と表示される（mpegts のフィールドレート表示）。

## 7. 再構成比較（B なし）

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 4 \
  -p 2 -q 27 --bframes 0 -b cuda --dump-recon build/work/rec.yuv \
  -o build/work/rec.264 -v 0
ffmpeg -v error -y -i build/work/rec.264 \
  -f rawvideo -pix_fmt yuv420p build/work/dec.yuv
cmp build/work/rec.yuv build/work/dec.yuv && echo SAME
```

## 8. B ありの再構成比較（コーディング順）

`--dump-recon` はコーディング順、FFmpeg の出力は表示順。`--dump-order` で並べ替える:

```sh
./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
  -p 2 -q 27 --bframes 3 -g 12 --b-pyramid 0 --weightb explicit \
  -b cuda --dump-recon build/work/b_rec.yuv --dump-order build/work/b_order.txt \
  -o build/work/b.264 -v 0
ffmpeg -v error -y -i build/work/b.264 \
  -f rawvideo -pix_fmt yuv420p build/work/b_dec.yuv
python3 - <<'PY'
w, h = 1920, 1080
fs = w * h * 3 // 2
order = [int(x) for x in open('build/work/b_order.txt')]
rec = open('build/work/b_rec.yuv', 'rb').read()
dec = open('build/work/b_dec.yuv', 'rb').read()
out = bytearray()
for k in order:
    out += dec[k * fs:(k + 1) * fs]
print('SAME' if bytes(out) == rec[:len(out)] else 'DIFF')
PY
```

期待値: `SAME`。

## 9. 10bit

```sh
./build/aegenc -i build/work/small.yuv -w 320 -h 192 -n 4 \
  --bit-depth 10 -p 2 -q 27 -o build/work/out10.264 -v 0
ffmpeg -v error -i build/work/out10.264 -f null - && echo DEC_OK
```

## 10. 回帰チェック一式

```sh
./build/backend_check
./build/api_check
./build/syntax_check
./build/kernel_check
./build/coef_check 200000
./build/cavlc_check 1000
./build/cabac_check 100000 8
./build/simd_check
./build/simd_check10
./build/nvrtc_check
```

期待値:
- `backend_check: 348 checks agree`
- `api_check: all checks passed`
- `coef_check` / `cavlc_check` / `cabac_check` / `simd_check` / `simd_check10`: exit 0
- `kernel_check: 9 kernels agree`
- `nvrtc_check`: CUDA kernels compile

## 11. 既知の未検証・不可

- 12bit（`--bit-depth 12`）: 既定バイナリでは明示エラー。`BIT_DEPTHS=12` の experimental ビルドでも decode 不能。
- lossless（`--lossless`）: 明示エラー。
- 4:2:2 / 4:4:4 / HEVC: 未実装。

## 12. 速度

```sh
for i in 1 2 3; do
  ./build/aegenc -i build/work/1080p.yuv -w 1920 -h 1080 -n 12 \
    -p 2 -q 27 --bframes 3 -g 12 -b cuda -o build/work/speed.264
done
```

3 回の中央値を使う。負荷依存が大きい（1.7〜2.0 s 帯）。
