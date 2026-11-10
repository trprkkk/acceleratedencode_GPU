# x264 fork (upstream source, not committed)

The upstream tree lives here locally but is intentionally excluded from git.
Only README_aeg_ja.md (fork notes), docs/FORK_PLAN_ja.md and tools/apply_x264_hint.py are tracked.

## Obtained

```
git clone --depth 1 https://code.videolan.org/videolan/x264.git forks/x264
# rev: 0480cb0 (2025-08-31, x264 0.165.x)
```

## Build recipe that works on this host (MinGW.org GCC 8.2, 32-bit only, no nasm)

```sh
cd forks/x264
CC=gcc ./configure --host=i686-pc-mingw32 --disable-asm --enable-static --enable-pic
mingw32-make -j4 "AR=ar rc " RANLIB=ranlib   # config.mak sets AR="gcc-ar rc "; gcc-ar fails here
./x264.exe --version                        # x264 0.165.x
```

Notes:
- The host gcc is 32-bit only and rejects -m64, so configure must be forced to i686.
- nasm is missing, so --disable-asm is required (slower, but A/B ratios stay meaningful).
- Always compare base vs patched with the SAME build, never against the packaged x264 binary.
