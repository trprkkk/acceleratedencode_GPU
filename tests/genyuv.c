/*
 * genyuv - deterministic synthetic I420 test clip.
 *
 * usage: genyuv <width> <height> <frames> <out.yuv> [mode]
 *
 * mode 0 (default) pans the whole frame, so every macroblock shares one motion
 * vector; mode 1 moves the top and the bottom half in opposite directions and
 * mode 2 does the same left/right, which is what makes 16x8 and 8x16 inter
 * partitions worth coding; mode 3 keeps the picture still after the first
 * frame so that P_Skip macroblocks get exercised; mode 4 moves the four
 * quadrants in four directions, which is what makes P_8x8 (one motion vector
 * per 8x8 block) worth coding.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
    int w = argc > 1 ? atoi(argv[1]) : 320;
    int h = argc > 2 ? atoi(argv[2]) : 192;
    int n = argc > 3 ? atoi(argv[3]) : 30;
    const char *path = argc > 4 ? argv[4] : "test.yuv";
    int mode = argc > 5 ? atoi(argv[5]) : 0;
    FILE *f = fopen(path, "wb");
    int t, x, y;

    if (!f) { perror(path); return 1; }
    for (t = 0; t < n; t++) {
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                int cx = x + t * 3, cy = y + t * 2;
                if (mode == 1) {
                    if (y < h / 2) { cx = x + t * 3; cy = y + t * 2; }
                    else           { cx = x - t * 3; cy = y - t * 2; }
                } else if (mode == 2) {
                    if (x < w / 2) { cx = x + t * 3; cy = y + t * 2; }
                    else           { cx = x - t * 3; cy = y - t * 2; }
                } else if (mode == 3) {
                    cx = x; cy = y;
                } else if (mode == 4) {
                    int sx = x < w / 2 ? 1 : -1, sy = y < h / 2 ? 1 : -1;
                    cx = x + sx * t * 3;
                    cy = y + sy * t * 2;
                }
                int v = 90 + (int)(60 * sin(cx * 0.09)) + (int)(40 * sin(cy * 0.13)) +
                        ((cx / 24 + cy / 24) & 1) * 30;
                if (v < 16) v = 16;
                if (v > 235) v = 235;
                fputc(v, f);
            }
        for (y = 0; y < h / 2; y++)
            for (x = 0; x < w / 2; x++)
                fputc(128 + (int)(40 * sin((x + (mode == 3 ? 0 : t)) * 0.05)), f);
        for (y = 0; y < h / 2; y++)
            for (x = 0; x < w / 2; x++)
                fputc(128 + (int)(40 * cos((y + (mode == 3 ? 0 : t)) * 0.07)), f);
    }
    fclose(f);
    printf("wrote %s (%dx%d, %d frames, mode %d)\n", path, w, h, n, mode);
    return 0;
}
