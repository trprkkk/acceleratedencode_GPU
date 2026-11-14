# x265 fork build notes (this host)

## Source
```
git clone https://bitbucket.org/multicoreware/x265_git.git forks/x265
git fetch --unshallow        # REQUIRED: cmake reads X265_LATEST_TAG from "git describe"
git describe --tags          # e.g. 4.2-59-gb81f650e2
```
Revision used: b81f650 (master, 2026-06-05).

## Configure (works with cmake 4.4.4 + MinGW.org gcc 8.2, 32-bit)
```
cd forks/x265 && mkdir build && cd build
cmake ../source -G "MinGW Makefiles" \
  -DENABLE_ASSEMBLY=OFF -DENABLE_CLI=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CXX_FLAGS=-include Q:/Downloads/HybridEncode2/repo/tools/x265_win_compat.h"
```
- CMAKE_POLICY_VERSION_MINIMUM=3.5 is needed because x265 tags 3.6/4.1 set CMP0025/CMP0054
  to OLD, which cmake 4 removed. Without it, configure dies on the policy errors.
- x265 4.1 (and 3.6/4.0) additionally fail on GROUP_AFFINITY even before that.

## Toolchain gaps that still block the build
MinGW.org GCC 8.2 headers lack the Windows 10 thread-group / NUMA API:
GROUP_AFFINITY, SetThreadGroupAffinity, GetThreadGroupAffinity,
GetNumaNodeProcessorMaskEx, GetNumaHighestNodeNumber.
tools/x265_win_compat.h shims those to no-ops and configure succeeds; the build then stops at:
```
c:\mingw\include\malloc.h:203: error: redefinition of 'void* x265::__mingw_aligned_malloc(size_t, size_t)'
```
i.e. x265 also collides with MinGW.org aligned-malloc helpers.

## Conclusion
A 32-bit MinGW.org-only host cannot build x265 as-is. Options:
1. Install an MSYS2 mingw-w64 toolchain (64-bit gcc + cmake) and build there.
2. Keep patching (shim __mingw_aligned_malloc / _aligned_malloc handling) - unknown number of further gaps.
