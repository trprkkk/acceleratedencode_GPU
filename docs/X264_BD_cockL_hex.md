# x264 hint BD-rate: cockL (hex hints)

Modes vs base, CRF 27, 3 reps.

| mode | CRF | time s | bytes | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| base | 27 | 103.618 | 504911 | 46.3547 | 0.991040 |
| limit:2 | 27 | 114.641 | 509231 | 46.3523 | 0.991037 |
| limit:4 | 27 | 116.215 | 510180 | 46.3604 | 0.991049 |
| seed | 27 | 115.714 | 508201 | 46.3411 | 0.991066 |
| skip | 27 | 112.854 | 500985 | 46.3506 | 0.991072 |

| mode | BD PSNR | BD SSIM | speed vs base |
|---|---:|---:|---:|
| limit:2 | +nan% | +nan% | 0.90x |
| limit:4 | +nan% | +nan% | 0.89x |
| seed | +nan% | +nan% | 0.90x |
| skip | +nan% | +nan% | 0.92x |
