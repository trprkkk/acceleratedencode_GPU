# Fuzzing

The fuzzers run the library API and the CLI's file parsers on arbitrary
input under AddressSanitizer and UndefinedBehaviorSanitizer. A crash, a
memory error, a leak, any UBSan report (signed overflow, a bad shift, an
out-of-range conversion) or a broken API contract counts as a finding. An
encoder that refuses an input with an error code is not a finding.

## Targets

| target | what it feeds | seeds |
|---|---|---|
| `fuzz_api` | Every `aeg_params` field: mostly in range, sometimes just outside it, sometimes wild. Then `aeg_encoder_open`. If the library accepts the parameters, up to six small pictures, the flush, and the rest of the API on the way (headers, `encode_image`, recon, the QP and QP-offset setters, the params and stats getters, `aeg_last_error`). Checks that every packet starts with a start code. | `corpus/api` |
| `fuzz_image` | `aeg_encoder_encode_image` with every colour space, depths 8..16, arbitrary strides and alignment, into 8- or 10-bit encoders, progressive or interlaced. Each plane is its own allocation of exactly stride × rows bytes, so ASan catches any read past the layout the caller described. If a layout breaks the documented contract, the call must return `AEG_EINVAL`. | `corpus/image` |
| `fuzz_cli` | The parsers in `aegenc` that read files: the YUV4MPEG2 demuxer (header and frames, all of its colour spaces and depths), the `--pass 2` statistics file (v1 and v2), and the `--fps` syntax. Their results go on into an encoder, the way `aegenc` uses them. `cli/aegenc.c` is included whole with `main` renamed, so the static parsers are tested as they are. The first input byte picks the parser. | `corpus/cli`, `fuzz_cli.dict` |

## Running

```sh
make fuzz                       # every target, 60 s each, gcc + standalone driver
make fuzz FUZZ_TIME=600         # longer
make fuzz FUZZ_ENGINE=libfuzzer CC=clang   # coverage-guided (libFuzzer)
```

`make fuzz-build` only builds `build-fuzz/fuzz_{api,image,cli}`. Its build
directory is separate, so the normal build is untouched. `make fuzz` copies
the seeds into `build-fuzz/corpus/<target>` (libFuzzer adds the inputs it
finds there) and stops at the first finding.  `FUZZ_CORPUS=dir` puts that
working corpus elsewhere, so it survives `rm -rf build-fuzz` and later runs
start from what earlier ones found.  `make fuzz-merge FUZZ_ENGINE=libfuzzer
CC=clang` shrinks it to the inputs that still add coverage (libFuzzer's
`-merge=1`).

Each binary takes libFuzzer's command line, with either engine:

```sh
./build-fuzz/fuzz_api tests/fuzz/corpus/api -max_total_time=600 -seed=7 \
    -artifact_prefix=/tmp/fz-api-
./build-fuzz/fuzz_cli build-fuzz/corpus/fuzz_cli -dict=tests/fuzz/fuzz_cli.dict
./build-fuzz/fuzz_api /tmp/fz-api-last-input       # reproduce one input
AEG_FUZZ_VERBOSE=1 ./build-fuzz/fuzz_api FILE      # print the decoded parameters
```

**Without clang** (`FUZZ_ENGINE=driver`, the default), `driver.c` stands in
for libFuzzer. It runs every corpus file, then random mutations of them
(bit flips, random and boundary bytes, inserted, deleted and duplicated runs, splices with other corpus files, dictionary tokens). It
has no coverage feedback, so it goes less deep than libFuzzer, but it catches
the same kinds of bug. The input being run is kept at
`<artifact_prefix>last-input`, so when a sanitizer aborts, the input that
caused it is still there. **With libFuzzer**, findings are written as
`crash-*`, `leak-*` and `timeout-*`.

Most noise comes from `fuzz_cli`: the parsers print their error messages for
malformed files. To see only the sanitizer output:
`... 2>&1 | grep -avE "^(warning|error|fuzz: bad)"`.

## CI

The `fuzz` job in `.github/workflows/ci.yml` runs every target for 120 s
under clang's libFuzzer. Its working corpus (`.ci/fuzz-corpus`) is kept in
the Actions cache: each run restores the latest one, merges the seeds in,
fuzzes, shrinks it with `make fuzz-merge` and saves it, so the 120 s add up
across runs instead of starting from the seeds every time. Then it runs 20 s
per target through the gcc driver, which keeps that build path working. If the job fails, the
reproducers are uploaded as the `fuzz-artifacts` artifact.

## Bugs found so far

All of these are fixed. None of the fixes changes the stream for in-range
parameters: the encodes are byte-identical to the previous build.

1. **VUI timing:** `time_scale = 2 * fps_num` overflowed `int` for large
   frame-rate numerators (`src/bitstream.c`).
2. **SAR reduction:** halving an over-large SAR as `(w + 1) / 2` overflowed
   at `INT_MAX` (`src/bitstream.c`).
3. **CLI `--fps`:** a tiny decimal such as `0.000125` parsed as the rate
   0/1001 instead of being refused (`parse_rate`, `cli/aegenc.c`).
4. **QP offsets:** `b_qp_offset` and `i_qp_offset` had no upper bound, and
   `cur_qp += offset` overflowed (`src/encoder.c`). Both are now clamped
   to 51.
5. **Two-pass statistics (found by review while writing `fuzz_cli`):** a v2
   file's index sized an allocation straight from the file. The parser
   (`stats_parse`) now bounds it by the file's size.

Getting the job running under clang found three more problems, none of
them in the encoder's output:

- **The tree did not build with clang 18.** `ENV_INT` (the
  environment-tuning macro in `src/encoder.c`) ended its statement
  expression with an `_Atomic int`, which clang ≤ 18 keeps atomic and then
  refuses in arithmetic. It now yields a plain `int`.
- **ASan's per-global COMDAT groups clash with the 10-bit build.** The
  10-bit objects are compiled from the same sources and only renamed
  (`_10`) afterwards. Clang names ASan's groups after a hash of the
  module's external symbols, which is the same for both objects, so GNU ld
  kept one and left the other's references in discarded sections. The
  libFuzzer build therefore adds
  `-fno-sanitize-address-globals-dead-stripping` (`FUZZ_CLANG` in the
  Makefile). Any other clang + ASan build of this tree needs it too.
- **libFuzzer's dictionary syntax is stricter than the driver's was:**
  `\n` is refused and libFuzzer exits. The dictionary now uses `\x0A`,
  and the driver refuses what libFuzzer refuses.

The fuzzing also corrected the documentation: odd widths and heights are
rounded down to even, and a caller's plane layout is checked against the
rounded size (`include/aeg/aeg.h`).
