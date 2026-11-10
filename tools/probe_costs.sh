#!/bin/bash
# Build the encoder's tq_batch kernel PTX with NVRTC and run the per-call cost
# probe against it.
set -e
cat > /tmp/aegmkptx3.c <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int nvrtcResult; typedef void *nvrtcProgram;
static int compile_one(const char *src, const char *name, const char *arch,
                       const char *out)
{
    void *lib = dlopen("libnvrtc.so", RTLD_NOW);
    nvrtcResult (*create)(nvrtcProgram *, const char *, const char *, int,
                          const char **, const char **);
    nvrtcResult (*compile)(nvrtcProgram, int, const char **);
    nvrtcResult (*getptxsize)(nvrtcProgram, size_t *);
    nvrtcResult (*getptx)(nvrtcProgram, char *);
    nvrtcResult (*destroy)(nvrtcProgram *);
    nvrtcResult (*getlogsize)(nvrtcProgram, size_t *);
    nvrtcResult (*getlog)(nvrtcProgram, char *);
    nvrtcProgram p = NULL;
    const char *opt = arch;
    size_t n = 0;
    char *ptx;
    FILE *f;
    create = dlsym(lib, "nvrtcCreateProgram");
    compile = dlsym(lib, "nvrtcCompileProgram");
    getptxsize = dlsym(lib, "nvrtcGetPTXSize");
    getptx = dlsym(lib, "nvrtcGetPTX");
    destroy = dlsym(lib, "nvrtcDestroyProgram");
    getlogsize = dlsym(lib, "nvrtcGetProgramLogSize");
    getlog = dlsym(lib, "nvrtcGetProgramLog");
    if (create(&p, src, name, 0, NULL, NULL)) return 1;
    if (compile(p, 1, &opt)) {
        getlogsize(p, &n);
        { char *log = malloc(n + 1); getlog(p, log); printf("%s\n", log); }
        return 1;
    }
    getptxsize(p, &n);
    ptx = malloc(n + 1);
    getptx(p, ptx);
    f = fopen(out, "wb");
    fwrite(ptx, 1, n, f);
    fclose(f);
    destroy(&p);
    return 0;
}
int main(int argc, char **argv)
{
    /* the encoder's own kernel: forward 4x4 transform + quantisation.
     * AEG_DEVFN becomes a plain static function, the OpenCL spellings the
     * CUDA ones -- exactly what src/backend_cuda.c's preamble does. */
    const char *pre =
        "typedef unsigned char uchar;\n"
        "typedef unsigned long long ulong;\n"
        "#define __kernel extern \"C\" __global__\n"
        "#define __global\n"
        "#define AEG_DEVFN static __device__\n"
        "#define get_global_id(d) ((int)(blockIdx.x * blockDim.x + threadIdx.x))\n";
    const char *k =
        "AEG_DEVFN int aeg_tq4(const int *r, int qbits, int f, int mf0, int mf1,\n"
        "                      int mf2, short *l)\n"
        "{\n"
        "    int i, t[16], d[16], nzc = 0;\n"
        "    for (i = 0; i < 4; i++) {\n"
        "        int d0 = r[i * 4 + 0], d1 = r[i * 4 + 1];\n"
        "        int d2 = r[i * 4 + 2], d3 = r[i * 4 + 3];\n"
        "        int t0 = d0 + d3, t1 = d1 + d2, t2 = d1 - d2, t3 = d0 - d3;\n"
        "        t[i * 4 + 0] = t0 + t1;\n"
        "        t[i * 4 + 1] = t3 * 2 + t2;\n"
        "        t[i * 4 + 2] = t0 - t1;\n"
        "        t[i * 4 + 3] = t3 - t2 * 2;\n"
        "    }\n"
        "    for (i = 0; i < 4; i++) {\n"
        "        int d0 = t[0 * 4 + i], d1 = t[1 * 4 + i];\n"
        "        int d2 = t[2 * 4 + i], d3 = t[3 * 4 + i];\n"
        "        int t0 = d0 + d3, t1 = d1 + d2, t2 = d1 - d2, t3 = d0 - d3;\n"
        "        d[0 * 4 + i] = t0 + t1;\n"
        "        d[1 * 4 + i] = t3 * 2 + t2;\n"
        "        d[2 * 4 + i] = t0 - t1;\n"
        "        d[3 * 4 + i] = t3 - t2 * 2;\n"
        "    }\n"
        "    for (i = 0; i < 16; i++) {\n"
        "        int v = d[(i & 3) * 4 + (i >> 2)];\n"
        "        int a = v < 0 ? -v : v;\n"
        "        int c = (i & 1) + ((i >> 2) & 1);\n"
        "        int mf = c == 0 ? mf0 : (c == 1 ? mf1 : mf2);\n"
        "        int lv = (a * mf + f) >> qbits;\n"
        "        l[i] = (short)(v < 0 ? -lv : lv);\n"
        "        nzc += (lv != 0);\n"
        "    }\n"
        "    return nzc;\n"
        "}\n"
        "__kernel void tq_batch(__global const short *res, int nblk, int qbits,\n"
        "                       int f, int mf0, int mf1, int mf2,\n"
        "                       __global short *level, __global int *nz)\n"
        "{\n"
        "    int b = get_global_id(0);\n"
        "    int i, r[16];\n"
        "    short l[16];\n"
        "    if (b >= nblk) return;\n"
        "    for (i = 0; i < 16; i++)\n"
        "        r[i] = res[16 * b + i];\n"
        "    nz[b] = aeg_tq4(r, qbits, f, mf0, mf1, mf2, l);\n"
        "    for (i = 0; i < 16; i++)\n"
        "        level[16 * b + i] = l[i];\n"
        "}\n";
    char *src = malloc(strlen(pre) + strlen(k) + 1);
    sprintf(src, "%s%s", pre, k);
    if (compile_one(src, "tq.cu", "--gpu-architecture=compute_60", argv[1]))
        return 1;
    printf("ptx written to %s\n", argv[1]);
    return 0;
}
EOF
gcc -O1 -o /tmp/aegmkptx3 /tmp/aegmkptx3.c -ldl
/tmp/aegmkptx3 /tmp/aegtq.ptx
gcc -O2 -o /tmp/aegcosts /mnt/q/Downloads/HybridEncode/costs_probe.c -ldl
export AEG_PTX=/tmp/aegtq.ptx
LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib /tmp/aegcosts
