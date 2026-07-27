cmake_minimum_required(VERSION 3.20)

project(
    CrisperWhisperCpp
    VERSION 1.1.1
    DESCRIPTION "Native C++ inference for CrisperWhisper 2.0"
    LANGUAGES C CXX
)

include(FetchContent)

option(CRISPERWHISPER_CUDA "Build the ggml CUDA backend" OFF)
option(CRISPERWHISPER_BUILD_TESTS "Build native tests" ON)
option(CRISPERWHISPER_BUILD_BENCHMARKS "Build the native benchmark runner" ON)
option(
    CRISPERWHISPER_STATIC_MSVC_RUNTIME
    "Statically link the Microsoft C/C++ runtime in release packages"
    OFF
)
set(
    CRISPERWHISPER_CPU_PROFILE
    "balance"
    CACHE STRING
    "CPU compatibility profile: portable, balance, or fast"
)
set_property(
    CACHE CRISPERWHISPER_CPU_PROFILE
    PROPERTY STRINGS portable balance fast
)
string(TOLOWER "${CRISPERWHISPER_CPU_PROFILE}" CRISPERWHISPER_CPU_PROFILE)

if(NOT CRISPERWHISPER_CPU_PROFILE MATCHES "^(portable|balance|fast)$")
    message(FATAL_ERROR
        "CRISPERWHISPER_CPU_PROFILE must be portable, balance, or fast"
    )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC AND CRISPERWHISPER_STATIC_MSVC_RUNTIME)
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
endif()

# Keep the dependency small: only the whisper library is needed.  The pin is
# deliberate because CrisperWhisper's converter writes whisper.cpp's legacy
# ggml model format and that format is part of our compatibility contract.
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${CRISPERWHISPER_CUDA} CACHE BOOL "" FORCE)

# Match the release profiles used by native ggml applications. CUDA builds
# also contain a CPU fallback, so the selected profile applies to both CPU-only
# and CUDA configurations.
set(_CRISPERWHISPER_GGML_SIMD_OPTIONS
    GGML_SSE42
    GGML_AVX
    GGML_AVX2
    GGML_AVX_VNNI
    GGML_BMI2
    GGML_AVX512
    GGML_AVX512_VBMI
    GGML_AVX512_VNNI
    GGML_AVX512_BF16
    GGML_FMA
    GGML_F16C
)

if(CRISPERWHISPER_CPU_PROFILE STREQUAL "fast")
    # Let ggml detect and use every instruction family available on the build
    # machine. The result should only be moved to equivalent CPUs.
    foreach(_option IN LISTS _CRISPERWHISPER_GGML_SIMD_OPTIONS)
        unset(${_option} CACHE)
    endforeach()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
elseif(CRISPERWHISPER_CPU_PROFILE STREQUAL "balance")
    # Stable AVX2 target for the large majority of modern x86-64 PCs.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "" FORCE)
    set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)
    if(NOT MSVC)
        set(GGML_FMA ON CACHE BOOL "" FORCE)
        set(GGML_F16C ON CACHE BOOL "" FORCE)
    endif()
    set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
else()
    # Baseline x86-64 build: avoid optional x86 SIMD instruction families.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_option IN LISTS _CRISPERWHISPER_GGML_SIMD_OPTIONS)
        set(${_option} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

message(STATUS
    "CrisperWhisper CPU profile: ${CRISPERWHISPER_CPU_PROFILE}"
)
# A static dependency graph makes the CLI a single executable on both
# Windows and Linux and avoids requiring users to copy ggml/whisper DLLs.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    whisper_cpp
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG 080bbbe85230f624f0b52127f1ae1218247989f9
    GIT_SHALLOW TRUE
    PATCH_COMMAND
        ${CMAKE_COMMAND}
        -DWHISPER_SOURCE_DIR=<SOURCE_DIR>
        -DWHISPER_PATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/patches/whisper-cross-attention.patch
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_whisper_patch.cmake
)
FetchContent_MakeAvailable(whisper_cpp)

add_library(crisperwhisper STATIC
    src/audio.cpp
    src/crisperwhisper.cpp
    src/word_timing.cpp
)

target_include_directories(crisperwhisper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${whisper_cpp_SOURCE_DIR}/examples
)

target_link_libraries(crisperwhisper PUBLIC whisper)

if(MSVC)
    target_compile_options(crisperwhisper PRIVATE /W4 /permissive-)
else()
    target_compile_options(crisperwhisper PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(crisper-whisper src/main.cpp)
target_link_libraries(crisper-whisper PRIVATE crisperwhisper)

include(GNUInstallDirs)
install(
    TARGETS crisper-whisper
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

set_target_properties(
    crisperwhisper crisper-whisper
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(CRISPERWHISPER_BUILD_TESTS)
    enable_testing()
    add_executable(crisperwhisper-tests tests/cpp/test_text.cpp)
    target_link_libraries(crisperwhisper-tests PRIVATE crisperwhisper)
    target_include_directories(
        crisperwhisper-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(
        crisperwhisper-tests
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME crisperwhisper-text COMMAND crisperwhisper-tests)
endif()

if(CRISPERWHISPER_BUILD_BENCHMARKS)
    add_executable(
        crisper-whisper-bench
        benchmarks/cpp_benchmark.cpp
    )
    target_link_libraries(crisper-whisper-bench PRIVATE crisperwhisper)
    set_target_properties(
        crisper-whisper-bench
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
