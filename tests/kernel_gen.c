/*
 * Writes the shared device kernel source (src/kernels.h) to stdout.
 *
 * The kernels live in a C string because they are handed to the OpenCL
 * compiler and to NVRTC at run time.  That makes them invisible to the host
 * compiler, so `make test` regenerates them into build/kernels_gen.c and
 * compiles that as C: the same text the GPU would see is then at least
 * syntax-checked, and tests/kernel_check.c runs it as ordinary code against
 * the CPU reference implementations.
 */
#include <stdio.h>

#include "kernels.h"

int main(void)
{
    fputs(aeg_kernel_src, stdout);
    return 0;
}
