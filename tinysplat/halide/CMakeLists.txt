cmake_minimum_required(VERSION 3.16)
project(tinysplat_halide_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Find Halide
# --------------------------------------------------------------------------
# HL_PATH can be set externally, default to $HOME/halide
if(NOT DEFINED HL_PATH)
    set(HL_PATH "$ENV{HL_PATH}")
endif()
if(NOT DEFINED HL_PATH)
    set(HL_PATH "$ENV{HOME}/halide")
endif()

if(NOT EXISTS "${HL_PATH}/lib/cmake/Halide/HalideConfig.cmake")
    message(FATAL_ERROR "Halide not found at ${HL_PATH}. "
        "Set HL_PATH environment variable or pass -DHL_PATH=/path/to/halide")
endif()

find_package(Halide REQUIRED CONFIG)
message(STATUS "Halide found: ${Halide_DIR}")
message(STATUS "Halide version: ${Halide_VERSION}")

# --------------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
set(SOURCES
    pipeline.cpp
)

# --------------------------------------------------------------------------
# Build as shared library (Python ctypes / PyBind11 loads this)
# --------------------------------------------------------------------------
add_library(tinysplat_halide_pipeline SHARED ${SOURCES})

target_include_directories(tinysplat_halide_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HL_PATH}/include
)

target_link_libraries(tinysplat_halide_pipeline PRIVATE
    Halide::Halide
    Halide::HalideUtils
    Threads::Threads
)

# OpenMP for CPU parallelism
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tinysplat_halide_pipeline PRIVATE OpenMP::OpenMP_CXX)
endif()

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
install(TARGETS tinysplat_halide_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES algorithm.h schedule_cpu.h DESTINATION include/tinysplat_halide)
