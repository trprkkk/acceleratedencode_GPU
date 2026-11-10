# GO gate: x265

1080p 120f, base vs all_off, 3 reps, interleaved order.
Primary KPI is the paired ratio; environment load is high.

| clip | preset | CRF | base s | all_off s | ratio |
|---|---|---:|---:|---:|---:|
| cockL | medium | 23 | 4.665 | 3.020 | **1.54x** |
| cockL | medium | 27 | 4.764 | 2.859 | **1.67x** |
| cockL | medium | 31 | 3.781 | 2.680 | **1.41x** |
| cockL | slow | 23 | 9.125 | 3.494 | **2.61x** |
| cockL | slow | 27 | 8.042 | 3.761 | **2.14x** |
| cockL | slow | 31 | 7.162 | 3.397 | **2.11x** |
| vtestL | medium | 23 | 6.999 | 5.445 | **1.29x** |
| vtestL | medium | 27 | 7.183 | 4.944 | **1.45x** |
| vtestL | medium | 31 | 6.795 | 4.418 | **1.54x** |
| vtestL | slow | 23 | 15.662 | 10.589 | **1.48x** |
| vtestL | slow | 27 | 11.285 | 5.094 | **2.22x** |
| vtestL | slow | 31 | 10.478 | 4.461 | **2.35x** |
| megaL | medium | 23 | 7.932 | 6.069 | **1.31x** |
| megaL | medium | 27 | 6.914 | 4.916 | **1.41x** |
| megaL | medium | 31 | 5.821 | 5.046 | **1.15x** |
| megaL | slow | 23 | 13.055 | 6.646 | **1.96x** |
| megaL | slow | 27 | 11.093 | 5.468 | **2.03x** |
| megaL | slow | 31 | 9.523 | 4.831 | **1.97x** |
| treeL | medium | 23 | 3.893 | 2.929 | **1.33x** |
| treeL | medium | 27 | 4.264 | 3.105 | **1.37x** |
| treeL | medium | 31 | 3.879 | 2.888 | **1.34x** |
| treeL | slow | 23 | 8.051 | 3.514 | **2.29x** |
| treeL | slow | 27 | 7.364 | 2.966 | **2.48x** |
| treeL | slow | 31 | 6.789 | 3.076 | **2.21x** |

## Gate summary

| preset | CRF | mean | worst |
|---|---:|---:|---:|
| medium | 23 | 1.37x | 1.29x |
| medium | 27 | 1.47x | 1.37x |
| medium | 31 | 1.36x | 1.15x |
| slow | 23 | 2.09x | 1.48x |
| slow | 27 | 2.22x | 2.03x |
| slow | 31 | 2.16x | 1.97x |

Gate: medium >= 1.15x, slow >= 1.50x

- medium CRF23: **PASS** (mean 1.37x, worst 1.29x)
- medium CRF27: **PASS** (mean 1.47x, worst 1.37x)
- medium CRF31: **PASS** (mean 1.36x, worst 1.15x)
- slow CRF23: **FAIL** (mean 2.09x, worst 1.48x)
- slow CRF27: **PASS** (mean 2.22x, worst 2.03x)
- slow CRF31: **PASS** (mean 2.16x, worst 1.97x)

Absolute times are wall-clock, high system load; use ratios.
