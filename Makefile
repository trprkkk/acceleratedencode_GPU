# aeg - accelerated encode GPU
#
#   make                  CPU build (no external dependencies)
#   make OPENCL=1         also build the OpenCL backend (loaded with dlopen,
#                         so no OpenCL headers or link-time library needed)
#   make CUDA=1           also build the CUDA backend (driver API + NVRTC,
#                         both loaded with dlopen -- no toolkit needed)
#   make test             encode a synthetic clip and verify the bitstream
#   make shared           also build the shared library (libaeg.so)
#   make install          install aegenc, libaeg.a, libaeg.so, aeg/aeg.h and
#                         aeg.pc under PREFIX (default /usr/local); DESTDIR
#                         stages it, LIBDIR/INCLUDEDIR/BINDIR override parts
#   make uninstall        remove what install put there
#   make examples         the example program, built against the public API
#   make BIT_DEPTHS=8     leave out the 10-bit (High 10) encoder
#
CC      ?= gcc
NVCC    ?= nvcc
# -O3 rather than -O2: the encoder is integer-only, so the aggressive
# vectoriser cannot change its output (verified byte-identical on every clip in
# tests/run_tests.sh and tools/sweep.sh), and it is worth ~13% of the overall
# speed -- 79.4 -> 89.4 fps at 320x192, QP 24, 60 frames, preset 2.
CFLAGS  ?= -O3 -g
CFLAGS  += -std=c11 -Wall -Wextra -Iinclude -Isrc -MMD -MP
LDFLAGS +=
LDLIBS  += -lm -ldl -lpthread

BUILD   := build
BIN     := $(BUILD)/aegenc
LIB     := $(BUILD)/libaeg.a

# The shared library: the same objects compiled position-independent and with
# every symbol hidden except the AEG_API ones (include/aeg/aeg.h).  The soname
# carries the major version; while it is 0 the ABI (struct layouts included)
# may change from one release to the next.
VERSION := $(shell sed -n 's/^\#define AEG_VERSION_STRING "\(.*\)"/\1/p' include/aeg/aeg.h)
SOMAJOR := $(firstword $(subst ., ,$(VERSION)))
SONAME  := libaeg.so.$(SOMAJOR)
SHLIB   := $(BUILD)/libaeg.so.$(VERSION)

PREFIX     ?= /usr/local
EXEC_PREFIX ?= $(PREFIX)
BINDIR     ?= $(EXEC_PREFIX)/bin
LIBDIR     ?= $(EXEC_PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
INSTALL    ?= install

# Bit depths, as x264 builds them: the sources that handle samples or
# coefficients (DEPTH_SRC) are compiled once per depth -- BIT_DEPTH=8 into
# $(BUILD)/, BIT_DEPTH=10 into $(BUILD)/obj10/ with every symbol they define
# renamed with a _10 suffix (objcopy), so both live in one library.  The rest
# (SHARED_SRC) is compiled once; api.c dispatches on aeg_params.bit_depth.
BIT_DEPTHS ?= 8 10
DEPTH_SRC  := src/transform.c src/intra.c src/mc.c src/deblock.c \
              src/cavlc.c src/cabac.c src/backend.c src/lowres.c \
              src/input.c src/encoder.c
SHARED_SRC := src/tables.c src/bitstream.c src/util.c src/lookahead.c \
              src/api.c
SRC := $(SHARED_SRC) $(DEPTH_SRC)
OBJ := $(patsubst %,$(BUILD)/%.o,$(basename $(notdir $(SRC))))
OBJCOPY ?= objcopy
NM      ?= nm
ifneq ($(filter 10,$(BIT_DEPTHS)),)
CFLAGS  += -DAEG_HAVE_10BIT
DEPTH_NAMES := $(basename $(notdir $(DEPTH_SRC)))
RAW10   := $(patsubst %,$(BUILD)/obj10/raw/%.o,$(DEPTH_NAMES))
OBJ10   := $(patsubst %,$(BUILD)/obj10/%.o,$(DEPTH_NAMES))
OBJ     += $(OBJ10)
endif

ifneq ($(filter 12,$(BIT_DEPTHS)),)
CFLAGS  += -DAEG_HAVE_12BIT
DEPTH_NAMES := $(basename $(notdir $(DEPTH_SRC)))
RAW12   := $(patsubst %,$(BUILD)/obj12/raw/%.o,$(DEPTH_NAMES))
OBJ12   := $(patsubst %,$(BUILD)/obj12/%.o,$(DEPTH_NAMES))
OBJ     += $(OBJ12)
endif

ifeq ($(OPENCL),1)
CFLAGS  += -DAEG_HAVE_OPENCL
SRC     += src/backend_opencl.c
OBJ     += $(BUILD)/backend_opencl.o
endif

ifeq ($(CUDA),1)
CFLAGS  += -DAEG_HAVE_CUDA
SRC     += src/backend_cuda.c
OBJ     += $(BUILD)/backend_cuda.o
endif

all: $(BIN)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# the 10-bit objects: compiled, then every symbol they define gets _10
# (references to the shared objects' symbols are left alone)
$(BUILD)/obj10/raw $(BUILD)/pic/obj10/raw:
	mkdir -p $@

$(BUILD)/obj10/raw/%.o: src/%.c | $(BUILD)/obj10/raw
	$(CC) $(CFLAGS) -DBIT_DEPTH=10 -c $< -o $@

$(BUILD)/obj10/syms: $(RAW10)
	$(NM) -A -P -g --defined-only $(RAW10) | \
	    awk '{ print $$2 " " $$2 "_10" }' | sort -u > $@

$(BUILD)/obj10/%.o: $(BUILD)/obj10/raw/%.o $(BUILD)/obj10/syms
	$(OBJCOPY) --redefine-syms=$(BUILD)/obj10/syms $< $@

ifneq ($(filter 12,$(BIT_DEPTHS)),)
$(BUILD)/obj12/raw:
	mkdir -p $@

$(BUILD)/obj12/raw/%.o: src/%.c | $(BUILD)/obj12/raw
	$(CC) $(CFLAGS) -DBIT_DEPTH=12 -c $< -o $@

$(BUILD)/obj12/syms: $(RAW12)
	$(NM) -A -P -g --defined-only $(RAW12) | \
	    awk '{ print $$2 " " $$2 "_12" }' | sort -u > $@

$(BUILD)/obj12/%.o: $(BUILD)/obj12/raw/%.o $(BUILD)/obj12/syms
	$(OBJCOPY) --redefine-syms=$(BUILD)/obj12/syms $< $@
endif

$(BUILD)/aegenc.o: cli/aegenc.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/libaeg.a: $(OBJ)
	$(AR) rcs $@ $(OBJ)

$(BIN): $(BUILD)/aegenc.o $(BUILD)/libaeg.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# ------------------------------------------------------------------ #
# Shared library, pkg-config file, install                           #
# ------------------------------------------------------------------ #
PIC_OBJ := $(patsubst $(BUILD)/%.o,$(BUILD)/pic/%.o,$(OBJ))

$(BUILD)/pic:
	mkdir -p $(BUILD)/pic

$(BUILD)/pic/%.o: src/%.c | $(BUILD)/pic
	$(CC) $(CFLAGS) -fPIC -fvisibility=hidden -c $< -o $@

PIC_RAW10 := $(patsubst $(BUILD)/%,$(BUILD)/pic/%,$(RAW10))

$(BUILD)/pic/obj10/raw/%.o: src/%.c | $(BUILD)/pic/obj10/raw
	$(CC) $(CFLAGS) -DBIT_DEPTH=10 -fPIC -fvisibility=hidden -c $< -o $@

$(BUILD)/pic/obj10/syms: $(PIC_RAW10)
	$(NM) -A -P -g --defined-only $(PIC_RAW10) | \
	    awk '{ print $$2 " " $$2 "_10" }' | sort -u > $@

$(BUILD)/pic/obj10/%.o: $(BUILD)/pic/obj10/raw/%.o $(BUILD)/pic/obj10/syms
	$(OBJCOPY) --redefine-syms=$(BUILD)/pic/obj10/syms $< $@

$(SHLIB): $(PIC_OBJ)
	$(CC) -shared -Wl,-soname,$(SONAME) $(LDFLAGS) $(PIC_OBJ) -o $@ $(LDLIBS)
	ln -sf libaeg.so.$(VERSION) $(BUILD)/$(SONAME)
	ln -sf libaeg.so.$(VERSION) $(BUILD)/libaeg.so

# The .pc file names the install paths, so `install` writes it afresh from
# the template with the PREFIX/LIBDIR of that run; $(BUILD)/aeg.pc is the
# same for the default paths.
PC_SED = sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@EXEC_PREFIX@|$(EXEC_PREFIX)|' \
	    -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	    -e 's|@VERSION@|$(VERSION)|' -e 's|@LIBS_PRIVATE@|$(LDLIBS)|' aeg.pc.in

$(BUILD)/aeg.pc: aeg.pc.in include/aeg/aeg.h Makefile | $(BUILD)
	$(PC_SED) > $@

shared: $(SHLIB)

install: $(BIN) $(LIB) $(SHLIB)
	$(INSTALL) -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBDIR) \
	    $(DESTDIR)$(INCLUDEDIR)/aeg $(DESTDIR)$(PKGCONFIGDIR)
	$(INSTALL) -m 755 $(BIN) $(DESTDIR)$(BINDIR)/aegenc
	$(INSTALL) -m 644 $(LIB) $(DESTDIR)$(LIBDIR)/libaeg.a
	$(INSTALL) -m 755 $(SHLIB) $(DESTDIR)$(LIBDIR)/libaeg.so.$(VERSION)
	ln -sf libaeg.so.$(VERSION) $(DESTDIR)$(LIBDIR)/$(SONAME)
	ln -sf libaeg.so.$(VERSION) $(DESTDIR)$(LIBDIR)/libaeg.so
	$(INSTALL) -m 644 include/aeg/aeg.h $(DESTDIR)$(INCLUDEDIR)/aeg/aeg.h
	$(PC_SED) > $(DESTDIR)$(PKGCONFIGDIR)/aeg.pc
	chmod 644 $(DESTDIR)$(PKGCONFIGDIR)/aeg.pc

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/aegenc $(DESTDIR)$(LIBDIR)/libaeg.a \
	    $(DESTDIR)$(LIBDIR)/libaeg.so.$(VERSION) \
	    $(DESTDIR)$(LIBDIR)/$(SONAME) $(DESTDIR)$(LIBDIR)/libaeg.so \
	    $(DESTDIR)$(INCLUDEDIR)/aeg/aeg.h $(DESTDIR)$(PKGCONFIGDIR)/aeg.pc
	-rmdir $(DESTDIR)$(INCLUDEDIR)/aeg

# The example uses nothing but include/aeg/aeg.h (tests/install_check.sh
# builds it again against an installed copy through pkg-config).
$(BUILD)/encode_example: examples/encode.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

examples: $(BUILD)/encode_example

# ------------------------------------------------------------------ #
# Tests                                                              #
# ------------------------------------------------------------------ #
$(BUILD)/genyuv: tests/genyuv.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/cavlc_check: tests/cavlc_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

$(BUILD)/syntax_check: tests/syntax_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

$(BUILD)/cabac_check: tests/cabac_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

$(BUILD)/cabac_mb_check: tests/cabac_mb_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

# The device kernels are a string that only the OpenCL runtime / NVRTC ever
# compiles.  Dump that string into a C file, then compile and *run* it against
# the CPU reference so a drifting kernel cannot hide.
$(BUILD)/kernel_gen: tests/kernel_gen.c src/kernels.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/kernels_gen.c: $(BUILD)/kernel_gen src/kernels.h
	./$(BUILD)/kernel_gen > $@

$(BUILD)/coef_check: tests/coef_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

$(BUILD)/kernel_check: tests/kernel_check.c $(BUILD)/kernels_gen.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) -I$(BUILD) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

# The device backends' host side (transfers, layouts, threads) on a real
# device against the CPU reference; skipped (and passing) without one.
$(BUILD)/backend_check: tests/backend_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

# The CUDA kernels through NVRTC (no GPU needed); skipped without NVRTC.
$(BUILD)/nvrtc_check: tests/nvrtc_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

# The public API on its own: logging, parameters, headers, strided input,
# timestamps, errors and encoders running in parallel threads.
$(BUILD)/api_check: tests/api_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

# What each batched device operation costs on the device next to the CPU
# reference (not run by `make test`: it is a measurement, not an assertion; see
# docs/DESIGN_ja.md).  `make OPENCL=1 CUDA=1 dev_bench` for both backends.
$(BUILD)/dev_bench: tests/dev_bench.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

dev_bench: $(BUILD)/dev_bench

# The input conversion (dither, chroma resampling, layouts) against
# references written in the test.
$(BUILD)/input_check: tests/input_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

# The SSE2/AVX2 kernels against scalar references, at 8 bits and (compiled
# with BIT_DEPTH=10, its library references renamed like the library's own
# 10-bit objects) at 10 bits.
$(BUILD)/simd_check: tests/simd_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

$(BUILD)/obj10/raw/simd_check.o: tests/simd_check.c | $(BUILD)/obj10/raw
	$(CC) $(CFLAGS) -DBIT_DEPTH=10 -c $< -o $@

$(BUILD)/simd_check10: $(BUILD)/obj10/raw/simd_check.o $(BUILD)/obj10/syms $(BUILD)/libaeg.a
	$(OBJCOPY) --redefine-syms=$(BUILD)/obj10/syms $< $(BUILD)/obj10/simd_check.o
	$(CC) $(CFLAGS) $(LDFLAGS) $(BUILD)/obj10/simd_check.o $(BUILD)/libaeg.a -o $@ $(LDLIBS)

SIMD_CHECKS := $(BUILD)/simd_check $(if $(filter 10,$(BIT_DEPTHS)),$(BUILD)/simd_check10)

# Install into a scratch DESTDIR, then build and run the example against it
# through pkg-config, shared and static, and check the exported symbols.
install-check: $(BIN) $(LIB) $(SHLIB) $(BUILD)/genyuv
	sh tests/install_check.sh $(BUILD)

# `make test` runs the table consistency check, the CAVLC round-trip check and
# the device-kernel cross-check (none of them need external dependencies), then
# the end-to-end clip check (needs an ffmpeg binary; set FFMPEG=... ).
test: $(BIN) $(BUILD)/genyuv $(BUILD)/cavlc_check $(BUILD)/syntax_check \
      $(BUILD)/cabac_check $(BUILD)/cabac_mb_check $(BUILD)/kernel_check \
      $(BUILD)/coef_check $(BUILD)/backend_check $(BUILD)/nvrtc_check \
      $(BUILD)/api_check $(BUILD)/input_check $(SIMD_CHECKS)
	for t in $(SIMD_CHECKS); do ./$$t && AEG_NOAVX2=1 ./$$t || exit 1; done
	./$(BUILD)/api_check
	./$(BUILD)/input_check
	./$(BUILD)/syntax_check
	./$(BUILD)/cavlc_check 2000
	./$(BUILD)/cabac_check 200000 8
	./$(BUILD)/cabac_mb_check 20000 1
	./$(BUILD)/kernel_check
	./$(BUILD)/backend_check
	./$(BUILD)/nvrtc_check
	./$(BUILD)/coef_check 200000
	sh tests/run_tests.sh

clean:
	rm -rf $(BUILD)

.PHONY: all test clean shared install uninstall examples install-check dev_bench
# every object's header dependencies (-MMD), the CLI's and the tests' too:
# a changed struct in aeg.h has to rebuild everything that includes it
-include $(OBJ:.o=.d) $(PIC_OBJ:.o=.d) $(wildcard $(BUILD)/*.d) \
         $(RAW10:.o=.d) $(PIC_RAW10:.o=.d)
