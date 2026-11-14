/* MinGW.org 8.2 compatibility shims for building x265 on this host.
 * The MinGW.org headers lack the Windows 10 thread-group / NUMA affinity API that
 * x265 uses, so provide minimal no-op definitions. Thread-group/NUMA pinning
 * degrades to a no-op on such toolchains.
 *
 * Usage (see forks/x265/README_build_ja.md):
 *   cmake ... "-DCMAKE_CXX_FLAGS=-include <abs path>/tools/x265_win_compat.h"
 */
#ifndef AEG_X265_WIN_COMPAT_H
#define AEG_X265_WIN_COMPAT_H
#ifdef _WIN32
#include <windows.h>
#include <string.h>

typedef struct _AEG_GROUP_AFFINITY {
    DWORD Mask;
    WORD  Group;
    WORD  Reserved[3];
} GROUP_AFFINITY, *PGROUP_AFFINITY;

static __inline BOOL SetThreadGroupAffinity(HANDLE h, const GROUP_AFFINITY *ga, GROUP_AFFINITY *prev)
{
    (void)h; (void)ga; (void)prev;
    return 0;
}

static __inline BOOL GetThreadGroupAffinity(HANDLE h, GROUP_AFFINITY *ga)
{
    (void)h;
    if (ga) { memset(ga, 0, sizeof(*ga)); ga->Mask = 1; }
    return 1;
}

static __inline WORD GetActiveProcessorGroupCount(void) { return 1; }

static __inline DWORD GetActiveProcessorCount(WORD g)
{
    SYSTEM_INFO si;
    (void)g;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}

static __inline BOOL GetNumaNodeProcessorMaskEx(USHORT node, GROUP_AFFINITY *ga)
{
    (void)node;
    if (ga) { memset(ga, 0, sizeof(*ga)); ga->Mask = 1; }
    return 0;
}

static __inline BOOL GetNumaHighestNodeNumber(PULONG n)
{
    if (n) *n = 0;
    return 0;
}
#endif
#endif
