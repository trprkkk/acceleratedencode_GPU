# GO gate: x264

1080p 120f, base vs all_off, 3 reps, interleaved order.
Primary KPI is the paired ratio; environment load is high.

| clip | preset | CRF | base s | all_off s | ratio |
|---|---|---:|---:|---:|---:|
| cockL | medium | 23 | 3.140 | 2.156 | **1.46x** |
| cockL | medium | 27 | 2.789 | 2.072 | **1.35x** |
| cockL | medium | 31 | 2.792 | 2.172 | **1.29x** |
| cockL | slow | 23 | 3.360 | 2.163 | **1.55x** |
| cockL | slow | 27 | 3.300 | 2.191 | **1.51x** |
| cockL | slow | 31 | 3.205 | 2.226 | **1.44x** |
| vtestL | medium | 23 | 2.944 | 2.079 | **1.42x** |
| vtestL | medium | 27 | 2.884 | 2.075 | **1.39x** |
| vtestL | medium | 31 | 2.800 | 2.054 | **1.36x** |
| vtestL | slow | 23 | 3.491 | 2.115 | **1.65x** |
| vtestL | slow | 27 | 3.216 | 2.095 | **1.53x** |
| vtestL | slow | 31 | 3.109 | 2.114 | **1.47x** |
| megaL | medium | 23 | 2.971 | 2.188 | **1.36x** |
| megaL | medium | 27 | 2.898 | 2.134 | **1.36x** |
| megaL | medium | 31 | 3.267 | 2.163 | **1.51x** |
| megaL | slow | 23 | 3.591 | 2.193 | **1.64x** |
| megaL | slow | 27 | 4.063 | 2.383 | **1.71x** |
| megaL | slow | 31 | 3.398 | 2.091 | **1.62x** |
| treeL | medium | 23 | 3.062 | 2.047 | **1.50x** |
| treeL | medium | 27 | 3.079 | 2.107 | **1.46x** |
| treeL | medium | 31 | 3.137 | 2.089 | **1.50x** |
| treeL | slow | 23 | 4.818 | 2.201 | **2.19x** |
| treeL | slow | 27 | 5.208 | 2.375 | **2.19x** |
| treeL | slow | 31 | 4.661 | 2.177 | **2.14x** |

## Gate summary

| preset | CRF | mean | worst |
|---|---:|---:|---:|
| medium | 23 | 1.43x | 1.36x |
| medium | 27 | 1.39x | 1.35x |
| medium | 31 | 1.42x | 1.29x |
| slow | 23 | 1.76x | 1.55x |
| slow | 27 | 1.73x | 1.51x |
| slow | 31 | 1.67x | 1.44x |

Gate: medium >= 1.20x, slow >= 1.40x

- medium CRF23: **PASS** (mean 1.43x, worst 1.36x)
- medium CRF27: **PASS** (mean 1.39x, worst 1.35x)
- medium CRF31: **PASS** (mean 1.42x, worst 1.29x)
- slow CRF23: **PASS** (mean 1.76x, worst 1.55x)
- slow CRF27: **PASS** (mean 1.73x, worst 1.51x)
- slow CRF31: **PASS** (mean 1.67x, worst 1.44x)

Absolute times are wall-clock, high system load; use ratios.
