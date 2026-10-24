# Comparing MV predictors against a reference decoder

The last reconstruction mismatch (P_8x8 sub-partitions, see `docs/STATUS.md`
bug 27) was invisible at the pixel level: one macroblock, thirty-eight luma
samples, from the third frame on.  What localised it in one run was comparing
*motion vectors*, not pixels.

## Encoder side (built in)

```sh
AEG_MBDBG=1 ./build/aegenc -w 320 -h 192 -q 23 -n 4 -i clip.yuv -o out.264 \
    --dump-recon rec.yuv 2> enc.log
```

prints, for every macroblock coded as P_8x8:

```
F3 MB 17,5 st 3333 [b0 mv -12,8 pmv -12,8] [b1 mv ... ] ...
```

`F` is the frame number, `MB x,y` the macroblock, `st` the four `sub_mb_type`
values, and one `[b<block> mv <dy> pmv <dy>]` group per partition in bitstream
order (`b` is the partition's first 4x4 block, raster order inside the
macroblock).

## Decoder side (not in this repository)

Take a scratch FFmpeg tree and print the pair inside the P_8x8 branch of
`decode_mb_cavlc()` in `libavcodec/h264_cavlc.c`, right after the mvd is read:

```c
pred_motion(h, sl, index, block_width, list, sl->ref_cache[list][scan8[index]], &mx, &my);
{ int pmvx = mx, pmvy = my;                       /* <- capture */
  mx += (unsigned)get_se_golomb(&sl->gb);
  my += (unsigned)get_se_golomb(&sl->gb);
  fprintf(stderr, "DEC mb %d,%d sub %d j %d pmv %d,%d mv %d,%d\n",
          sl->mb_x, sl->mb_y, i, j, pmvx, pmvy, mx, my); }
```

plus a per-picture counter printed at the top of `decode_slice()`
(`libavcodec/h264_slice.c`), and decode with `-threads 1` so the order matches.

## Diffing

Align the two listings per macroblock and per partition.  The `mvd`
(`mv - pmv`) is identical on both sides, so any line where the encoder's `pmv`
differs from the decoder's is a predictor bug -- not a bitstream bug.  In bug
27 the encoder predicted from a 4x4 block that the decoder had not filled in
yet (`PART_NOT_AVAILABLE`), so the decoder substituted the above-left block
instead.
