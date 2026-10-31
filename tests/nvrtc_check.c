/*
 * The CUDA kernels through NVRTC, without a GPU.
 *
 * The CUDA backend compiles src/kernels.h at run time with NVRTC, behind a
 * preamble that maps the OpenCL spelling onto CUDA's.  Nothing in a build
 * without a GPU ever compiles that combination, so a kernel edit that is valid
 * OpenCL C but not valid CUDA C++ (or a preamble that stops covering a new
 * construct) would only show on the first machine with a GPU -- as a silent
 * fallback to the CPU.  NVRTC itself needs no GPU and no driver, so this
 * compiles the exact source, preamble and options the backend uses, for every
 * architecture the NVRTC at hand supports, and checks that every kernel the
 * backend looks up is in the PTX.  (Whether the kernels compute the right
 * thing is tests/kernel_check.c's and tests/backend_check.c's job.)
 *
 * NVRTC without the CUDA toolkit: pip install nvidia-cuda-nvrtc-cu12, then
 * AEG_NVRTC_LIB=<site-packages>/nvidia/cuda_nvrtc/lib/libnvrtc.so.12.
 * Without NVRTC, or in a build without CUDA=1, it says so and passes.
 *
 *   nvrtc_check [cc]          (default: every supported architecture)
 */
#include <stdio.h>
#include <stdlib.h>

#include "backend.h"

int main(int argc, char **argv)
{
    char log[4096];
    size_t ptx = 0;
    int cc = argc > 1 ? atoi(argv[1]) : 0;
    int rc = aeg_cuda_compile_check(cc, log, sizeof(log), &ptx);
    if (rc == AEG_EBACKEND) {
        printf("   nvrtc_check: no NVRTC (or no CUDA=1 build), skipped\n");
        return 0;
    }
    if (rc != AEG_OK) {
        printf("   nvrtc_check: FAILED\n%s\n", log);
        return 1;
    }
    printf("   nvrtc_check: the CUDA kernels compile (%s; last PTX %zu bytes)\n",
           cc ? argv[1] : "every architecture NVRTC supports", ptx);
    return 0;
}
