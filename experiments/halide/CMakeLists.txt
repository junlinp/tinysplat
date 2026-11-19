cmake_minimum_required(VERSION 3.16)
project(tinysplat_halide_pipeline LANGUAGES CXX)

find_package(Halide REQUIRED CONFIG
    HINTS "${HL_PATH}" "$ENV{HL_PATH}" "$ENV{HOME}/halide"
    PATH_SUFFIXES lib64/cmake/Halide
)
find_package(Threads REQUIRED)
find_package(OpenMP)

add_library(tinysplat_halide_pipeline SHARED
    pipeline.cpp
    algorithm.cpp
)
set_target_properties(tinysplat_halide_pipeline PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    INSTALL_RPATH_USE_LINK_PATH YES
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tinysplat_halide_pipeline PRIVATE
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
    )
endif()

target_link_libraries(tinysplat_halide_pipeline PRIVATE Halide::Halide Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tinysplat_halide_pipeline PRIVATE OpenMP::OpenMP_CXX)
endif()

include(CTest)
if(BUILD_TESTING)
    foreach(check IN ITEMS forward backward)
        add_executable(halide_test_${check} test_${check}.cpp)
        target_compile_features(halide_test_${check} PRIVATE cxx_std_17)
        target_link_libraries(halide_test_${check} PRIVATE tinysplat_halide_pipeline Halide::Halide)
        add_test(NAME halide_${check} COMMAND halide_test_${check})
    endforeach()
endif()

install(TARGETS tinysplat_halide_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES algorithm.h schedule_cpu.h DESTINATION include/tinysplat_halide)
