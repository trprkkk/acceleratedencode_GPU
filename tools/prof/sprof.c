/* Minimal sampling profiler: LD_PRELOAD=tools/prof/sprof.so ./build/aegenc ...
 * SIGPROF every 1 ms of CPU time; each sample records the thread (main or
 * not) and the frame-pointer call chain.  Needs a build with
 * -fno-omit-frame-pointer.  Writes $SPROF_OUT (default /tmp/sprof.out);
 * tools/prof/an.py turns it into self / inclusive tables. */
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

#define MAXS 400000
#define DEPTH 48
static uintptr_t buf[MAXS][DEPTH + 2];
static volatile int n;
static pid_t mainpid;

static void handler(int sig, siginfo_t *si, void *uc_)
{
    ucontext_t *uc = (ucontext_t *)uc_;
    int k = __sync_fetch_and_add(&n, 1), d = 0;
    uintptr_t *fp, *s, lo, hi;
    (void)sig; (void)si;
    if (k >= MAXS)
        return;
    s = buf[k];
    s[0] = (uintptr_t)(syscall(SYS_gettid) == mainpid);
    s[2 + d++] = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
    fp = (uintptr_t *)uc->uc_mcontext.gregs[REG_RBP];
    /* frames live on the interrupted stack: anything outside [rsp, rsp + 512K)
     * is a general-purpose rbp (code without frame pointers), stop there */
    lo = (uintptr_t)uc->uc_mcontext.gregs[REG_RSP];
    hi = lo + (512u << 10);
    while (d < DEPTH && (uintptr_t)fp >= lo && (uintptr_t)fp + 16 <= hi &&
           ((uintptr_t)fp & 7) == 0) {
        uintptr_t ret = fp[1], *nfp = (uintptr_t *)fp[0];
        if (!ret || nfp <= fp)
            break;
        s[2 + d++] = ret - 1;
        fp = nfp;
    }
    s[1] = (uintptr_t)d;
}

static void dump(void)
{
    const char *o = getenv("SPROF_OUT");
    FILE *f = fopen(o ? o : "/tmp/sprof.out", "w");
    FILE *m = fopen("/proc/self/maps", "r");
    char line[512];
    int i, j, cnt = n < MAXS ? n : MAXS;
    struct itimerval z;
    memset(&z, 0, sizeof(z));
    setitimer(ITIMER_PROF, &z, NULL);
    if (!f)
        return;
    while (m && fgets(line, sizeof(line), m))
        if (strstr(line, " r-xp ") || strstr(line, " r--p 00000000"))
            fprintf(f, "M %s", line);
    if (m)
        fclose(m);
    for (i = 0; i < cnt; i++) {
        fprintf(f, "S %d", (int)buf[i][0]);
        for (j = 0; j < (int)buf[i][1]; j++)
            fprintf(f, " %lx", (unsigned long)buf[i][2 + j]);
        fputc('\n', f);
    }
    fclose(f);
}

__attribute__((constructor)) static void init(void)
{
    struct sigaction sa;
    struct itimerval it;
    mainpid = getpid();
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGPROF, &sa, NULL);
    it.it_interval.tv_sec = 0;
    it.it_interval.tv_usec = 1000;
    it.it_value = it.it_interval;
    setitimer(ITIMER_PROF, &it, NULL);
    atexit(dump);
}
