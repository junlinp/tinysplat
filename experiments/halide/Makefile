# Makefile for TinySplat Halide Pipeline
# =======================================
# Requires: Halide v16+ (https://halide-lang.org)
#
# Build:
#   HL_PATH=/path/to/halide make
#
#   Produces: libtinysplat_halide_pipeline.so

HL_PATH ?= $(HOME)/halide
CXX    ?= g++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -fPIC \
           -I$(HL_PATH)/include \
           -I$(HL_PATH)/share/Halide/tools \
           -I.
LDFLAGS = -L$(HL_PATH)/lib -lHalide -lpthread -ldl \
          -Wl,-rpath,$(HL_PATH)/lib

SRC_DIR  = .
BUILD_DIR = build

SOURCES = $(SRC_DIR)/pipeline.cpp $(SRC_DIR)/algorithm.cpp

TARGET = libtinysplat_halide_pipeline.so

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -o $(BUILD_DIR)/$@ $(SOURCES) $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

# Alias for in-tree build
lib: $(TARGET)

# Install
PREFIX ?= /usr/local
install: $(TARGET)
	install -d $(PREFIX)/lib
	install -m 755 $(BUILD_DIR)/$(TARGET) $(PREFIX)/lib/
	install -d $(PREFIX)/include/tinysplat_halide
	install -m 644 $(SRC_DIR)/algorithm.h $(SRC_DIR)/schedule_cpu.h $(PREFIX)/include/tinysplat_halide/
