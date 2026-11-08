/* A standalone runner for the fuzz targets, for compilers without libFuzzer
 * (gcc).  Built with ASan + UBSan it finds the same class of bugs, only
 * without coverage feedback: inputs are random mutations of the corpus.
 *
 *   driver [options] [file-or-dir ...]
 *
 * Files are run once each.  Directories are a corpus: every file in them is
 * run, then mutated.  Options (libFuzzer's names, so the same command line
 * works for both builds):
 *   -runs=N            mutated inputs to run (default 0: only the files)
 *   -max_total_time=S  stop mutating after S seconds
 *   -max_len=N         longest input generated (default 4096)
 *   -seed=N            random seed (default 1)
 *   -dict=FILE         libFuzzer dictionary: "token" per line (\n \" \\ \xNN)
 *   -artifact_prefix=P the input being run is kept at P + "last-input", so a
 *                      sanitizer abort leaves the input that caused it
 */
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "fuzz.h"

typedef struct {
    uint8_t *p;
    size_t n;
} blob;

static blob *corpus, *dict;
static size_t ncorpus, ndict;
static uint64_t rng = 1;
static char last_path[1024] = "last-input";

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng >> 11);
}

static void add(blob **v, size_t *n, const uint8_t *p, size_t len)
{
    blob *nv = (blob *)realloc(*v, (*n + 1) * sizeof(blob));
    if (!nv)
        exit(1);
    *v = nv;
    nv[*n].p = (uint8_t *)malloc(len ? len : 1);
    if (!nv[*n].p)
        exit(1);
    memcpy(nv[*n].p, p, len);
    nv[*n].n = len;
    (*n)++;
}

static int read_file(const char *path, uint8_t **p, size_t *n)
{
    FILE *f = fopen(path, "rb");
    long len;
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    *p = (uint8_t *)malloc(len > 0 ? (size_t)len : 1);
    *n = len > 0 && *p ? fread(*p, 1, (size_t)len, f) : 0;
    fclose(f);
    return *p ? 0 : -1;
}

static void run(const uint8_t *p, size_t n)
{
    /* keep the input on disk first: a sanitizer report aborts the process */
    FILE *f = fopen(last_path, "wb");
    uint8_t *copy = (uint8_t *)malloc(n ? n : 1);
    if (f) {
        fwrite(p, 1, n, f);
        fclose(f);
    }
    if (!copy)
        exit(1);
    memcpy(copy, p, n);         /* an exact-size buffer, for ASan */
    LLVMFuzzerTestOneInput(copy, n);
    free(copy);
}

static void load_dict(const char *path)
{
    uint8_t *p;
    size_t n, i = 0;
    if (read_file(path, &p, &n) < 0) {
        fprintf(stderr, "driver: cannot read %s\n", path);
        exit(1);
    }
    while (i < n) {
        uint8_t tok[256];
        size_t t = 0;
        while (i < n && p[i] != '"' && p[i] != '\n' && p[i] != '#')
            i++;
        if (i < n && p[i] == '#') {
            while (i < n && p[i] != '\n')
                i++;
            continue;
        }
        if (i >= n || p[i] != '"') {
            i++;
            continue;
        }
        for (i++; i < n && p[i] != '"' && t < sizeof(tok); i++) {
            if (p[i] == '\\' && i + 1 < n) {
                i++;
                if (p[i] == 'n')
                    tok[t++] = '\n';
                else if (p[i] == 'x' && i + 2 < n) {
                    char hex[3] = {(char)p[i + 1], (char)p[i + 2], 0};
                    tok[t++] = (uint8_t)strtol(hex, NULL, 16);
                    i += 2;
                } else
                    tok[t++] = p[i];
            } else
                tok[t++] = p[i];
        }
        i++;
        if (t)
            add(&dict, &ndict, tok, t);
    }
    free(p);
}

static size_t mutate(uint8_t *buf, size_t n, size_t max)
{
    int k, rounds = 1 + (int)(rnd() % 4);
    for (k = 0; k < rounds; k++) {
        size_t pos = n ? rnd() % n : 0, len;
        switch (rnd() % 9) {
        case 0:                                     /* flip a bit */
            if (n)
                buf[pos] ^= (uint8_t)(1u << (rnd() % 8));
            break;
        case 1:                                     /* a random byte */
            if (n)
                buf[pos] = (uint8_t)rnd();
            break;
        case 2: {                                   /* an interesting byte */
            static const uint8_t v[] = {0, 1, 0x7f, 0x80, 0xff, '0', '9', ' ',
                                        '\n', '-'};
            if (n)
                buf[pos] = v[rnd() % sizeof(v)];
            break;
        }
        case 3:                                     /* insert a byte */
            if (n < max) {
                memmove(buf + pos + 1, buf + pos, n - pos);
                buf[pos] = (uint8_t)rnd();
                n++;
            }
            break;
        case 4:                                     /* delete a run */
            if (n) {
                len = 1 + rnd() % (n - pos < 16 ? n - pos : 16);
                memmove(buf + pos, buf + pos + len, n - pos - len);
                n -= len;
            }
            break;
        case 5:                                     /* duplicate a run */
            if (n) {
                len = 1 + rnd() % (n - pos < 32 ? n - pos : 32);
                if (n + len <= max) {
                    memmove(buf + pos + len, buf + pos, n - pos);
                    n += len;
                }
            }
            break;
        case 6:                                     /* splice a corpus entry */
            if (ncorpus) {
                const blob *b = &corpus[rnd() % ncorpus];
                size_t from = b->n ? rnd() % b->n : 0;
                len = b->n - from;
                if (pos + len > max)
                    len = max - pos;
                memcpy(buf + pos, b->p + from, len);
                if (pos + len > n)
                    n = pos + len;
            }
            break;
        case 7:                                     /* a dictionary token */
            if (ndict) {
                const blob *b = &dict[rnd() % ndict];
                if (n + b->n <= max) {
                    if (rnd() & 1) {                /* insert */
                        memmove(buf + pos + b->n, buf + pos, n - pos);
                        n += b->n;
                    } else if (pos + b->n > n) {    /* overwrite */
                        n = pos + b->n;
                    }
                    memcpy(buf + pos, b->p, b->n);
                }
            }
            break;
        default:                                    /* truncate */
            n = pos;
            break;
        }
    }
    return n;
}

int main(int argc, char **argv)
{
    long runs = 0, max_time = 0, max_len = 4096, i, done = 0;
    time_t t0 = time(NULL);
    uint8_t *buf;
    int a;

    for (a = 1; a < argc; a++) {
        const char *s = argv[a];
        if (!strncmp(s, "-runs=", 6))
            runs = atol(s + 6);
        else if (!strncmp(s, "-max_total_time=", 16))
            max_time = atol(s + 16);
        else if (!strncmp(s, "-max_len=", 9))
            max_len = atol(s + 9);
        else if (!strncmp(s, "-seed=", 6))
            rng = (uint64_t)atoll(s + 6) * 2654435761u + 1;
        else if (!strncmp(s, "-dict=", 6))
            load_dict(s + 6);
        else if (!strncmp(s, "-artifact_prefix=", 17))
            snprintf(last_path, sizeof(last_path), "%slast-input", s + 17);
        else if (s[0] == '-')
            fprintf(stderr, "driver: ignoring %s\n", s);
        else {
            struct stat stbuf;
            uint8_t *p;
            size_t n;
            if (stat(s, &stbuf) == 0 && S_ISDIR(stbuf.st_mode)) {
                DIR *d = opendir(s);
                struct dirent *de;
                while (d && (de = readdir(d))) {
                    char path[2048];
                    if (de->d_name[0] == '.')
                        continue;
                    snprintf(path, sizeof(path), "%s/%s", s, de->d_name);
                    if (read_file(path, &p, &n) == 0) {
                        add(&corpus, &ncorpus, p, n);
                        free(p);
                    }
                }
                if (d)
                    closedir(d);
            } else if (read_file(s, &p, &n) == 0) {
                run(p, n);
                free(p);
                done++;
            } else {
                fprintf(stderr, "driver: cannot read %s\n", s);
                return 1;
            }
        }
    }
    for (i = 0; i < (long)ncorpus; i++, done++)
        run(corpus[i].p, corpus[i].n);
    if (max_len < 1)
        max_len = 1;
    buf = (uint8_t *)malloc((size_t)max_len);
    if (!buf)
        return 1;
    for (i = 0; i < runs || (max_time > 0 && runs == 0); i++, done++) {
        size_t n = 0;
        if (max_time > 0 && time(NULL) - t0 >= max_time)
            break;
        if (ncorpus && (rnd() % 8)) {
            const blob *b = &corpus[rnd() % ncorpus];
            n = b->n < (size_t)max_len ? b->n : (size_t)max_len;
            memcpy(buf, b->p, n);
        } else {                        /* now and then: fresh random bytes */
            size_t j;
            n = rnd() % (max_len < 512 ? (size_t)max_len : 512);
            for (j = 0; j < n; j++)
                buf[j] = (uint8_t)rnd();
        }
        n = mutate(buf, n, (size_t)max_len);
        run(buf, n);
    }
    fprintf(stderr, "driver: %ld inputs in %lds, no crash\n", done,
            (long)(time(NULL) - t0));
    remove(last_path);
    free(buf);
    return 0;
}
