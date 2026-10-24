# aeg - accelerated encode GPU
#
#   make                  CPU build (no external dependencies)
#   make OPENCL=1         also build the OpenCL backend (loaded with dlopen,
#                         so no OpenCL headers or link-time library needed)
#   make CUDA=1           also build the CUDA backend (requires nvcc)
#   make test             encode a synthetic clip and verify the bitstream
#
CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -g
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
LDLIBS  += -lcudart -lnvrtc
SRC     += src/backend_cuda.cu
OBJ     += $(BUILD)/backend_cuda.o
$(BUILD)/backend_cuda.o: src/backend_cuda.cu
	$(NVCC) -O2 -c $< -o $@
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

# `make test` runs the table consistency check, the CAVLC round-trip check
# (both without external dependencies) and the end-to-end clip check (needs an
# ffmpeg binary; set FFMPEG=... ).
test: $(BIN) $(BUILD)/genyuv $(BUILD)/cavlc_check $(BUILD)/syntax_check
	./$(BUILD)/syntax_check
	./$(BUILD)/cavlc_check 2000
	sh tests/run_tests.sh

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
-include $(OBJ:.o=.d)
