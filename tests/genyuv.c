/*
 * genyuv - deterministic synthetic I420 test clip.
 *
 * usage: genyuv <width> <height> <frames> <out.yuv>
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
    FILE *f = fopen(path, "wb");
    int t, x, y;

    if (!f) { perror(path); return 1; }
    for (t = 0; t < n; t++) {
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                int cx = x + t * 3, cy = y + t * 2;
                int v = 90 + (int)(60 * sin(cx * 0.09)) + (int)(40 * sin(cy * 0.13)) +
                        ((cx / 24 + cy / 24) & 1) * 30;
                if (v < 16) v = 16;
                if (v > 235) v = 235;
                fputc(v, f);
            }
        for (y = 0; y < h / 2; y++)
            for (x = 0; x < w / 2; x++)
                fputc(128 + (int)(40 * sin((x + t) * 0.05)), f);
        for (y = 0; y < h / 2; y++)
            for (x = 0; x < w / 2; x++)
                fputc(128 + (int)(40 * cos((y + t) * 0.07)), f);
    }
    fclose(f);
    printf("wrote %s (%dx%d, %d frames)\n", path, w, h, n);
    return 0;
}
