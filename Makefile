# aeg - accelerated encode GPU
#
#   make                  CPU build (no external dependencies)
#   make OPENCL=1         also build the OpenCL backend (loaded with dlopen,
#                         so no OpenCL headers or link-time library needed)
#   make CUDA=1           also build the CUDA backend (driver API + NVRTC,
#                         both loaded with dlopen -- no toolkit needed)
#   make test             encode a synthetic clip and verify the bitstream
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

SRC := src/tables.c src/transform.c src/intra.c src/mc.c src/deblock.c \
       src/bitstream.c src/cavlc.c src/util.c src/backend.c src/encoder.c
OBJ := $(patsubst %,$(BUILD)/%.o,$(basename $(notdir $(SRC))))

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

$(BUILD)/aegenc.o: cli/aegenc.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/libaeg.a: $(OBJ)
	$(AR) rcs $@ $(OBJ)

$(BIN): $(BUILD)/aegenc.o $(BUILD)/libaeg.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# ------------------------------------------------------------------ #
# Tests                                                              #
# ------------------------------------------------------------------ #
$(BUILD)/genyuv: tests/genyuv.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/cavlc_check: tests/cavlc_check.c $(BUILD)/libaeg.a | $(BUILD)
	$(CC) $(CFLAGS) $< $(BUILD)/libaeg.a -o $@ $(LDLIBS)

$(BUILD)/syntax_check: tests/syntax_check.c $(BUILD)/libaeg.a | $(BUILD)
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

# `make test` runs the table consistency check, the CAVLC round-trip check and
# the device-kernel cross-check (none of them need external dependencies), then
# the end-to-end clip check (needs an ffmpeg binary; set FFMPEG=... ).
test: $(BIN) $(BUILD)/genyuv $(BUILD)/cavlc_check $(BUILD)/syntax_check $(BUILD)/kernel_check $(BUILD)/coef_check
	./$(BUILD)/syntax_check
	./$(BUILD)/cavlc_check 2000
	./$(BUILD)/kernel_check
	./$(BUILD)/coef_check 200000
	sh tests/run_tests.sh

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
-include $(OBJ:.o=.d)
