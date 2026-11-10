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
