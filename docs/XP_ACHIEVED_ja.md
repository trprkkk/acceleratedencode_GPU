# 案C 達成パレート点（640p / x264 slow / GPU hints / limit:4）

## 達成内容

| 項目 | 値 |
|---|---|
| プリセット | x264 **slow** |
| 解像度 | 640p（cockL / vtestL / megaL / treeL） |
| ヒント | GPU full-res 階層探索（range8、候補2、SAD 選択） |
| limit | **4** |
| 生成 | **2.55 s / 240f（10.6 ms/frame、3回中央値 2.37–2.62）** |
| E2E | **4/4 で base 以下**（treeL は autoskip で base 経路） |
| BD-rate | 平均 **-0.117% PSNR / -0.107% SSIM** |

## クリップ別

| clip | 生成 | E2E | base | 出力差 | PSNR BD | SSIM BD |
|---|---:|---:|---:|---:|---:|---:|
| cockL | 2.55 s | 13.01 s | 14.51 s | +0.11% | -0.097% | -0.175% |
| vtestL | 1.5 s | 8.1 s | 9.66 s | -0.8% | -0.388% | -0.382% |
| megaL | 2.0 s | 7.1 s | 7.55 s | -0.3% | +0.017% | +0.131% |
| treeL | autoskip | 3.73 s | 3.73 s | ±0% | 0.000% | 0.000% |

- **4/4**: treeL は autoskip により base 経路（E2E 同等）。
- 出力差はすべて base ±1%。
- BD-rate は 4クリップ × CRF23/27/31。

## autoskip

`tools/x264_hint_autoskip.sh`:
1. 10 フレーム base エンコードで `base_ms/frame` を推定。
2. 10 フレームのヒント生成で `gen_ms/frame` を推定。
3. `gen_ms > 0.3 * base_ms` ならヒント生成せず base 経路。

実測（treeL）:
- base 10f: **24.16 ms/frame**
- gen 10f: **137.52 ms/frame**
- 閾値 7.25 ms → **skip = 1**。

他クリップは gen < 0.3×base でヒント経路。

## 未達

- **1080p**: `cuda_me_batch` 内の pinned 宛先 `memcpy` で SIGSEGV
  （フレーム0のみ）。凍結。CPU フォールバックは 66.7 ms/frame。
- **medium**: E2E 勝ち点なし。
- treeL はヒント経路では負け（autoskip 運用で回避）。

## 次段の提案

1. **GPU 解析を先読みコスト側へ移す**（mbtree / slicetype の SATD 集計）。
   ME ヒントは slow でのみ E2E 勝ち。先読みは medium でも効く可能性。
2. 1080p 再挑戦時:
   - `cuda_me_batch` を使わず、`sad_batch` 単発で 1 MB 以下に分割。
   - pinned 宛先ではなく **デバイス常駐 + `me_pic_upload`** のみ。
   - フレーム0→1 の境界で `cuda_buf` のサイズを照合。
3. x265 アダプタ（slow 相当のパレート改善）。
