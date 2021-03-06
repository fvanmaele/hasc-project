cmake_minimum_required (VERSION 3.12)
project(hasc-project-vanmaele)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(benchmark CONFIG REQUIRED)
#find_package(TBB REQUIRED)
#find_package(vectorclass REQUIRED)

# Bundled libraries
set(OPT_USE_BUNDLED_BENCHMARK ON)

if (OPT_USE_BUNDLED_BENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_INSTALL OFF)
    add_subdirectory("benchmark")
else()
    find_package(benchmark CONFIG REQUIRED)
endif()

# Define optimization levels
set(OPT_VEC_REPORT OFF)
set(OPT_FAST_MATH OFF)
set(OPT_TSAN OFF)
set(OPT_ASAN OFF)

# Vectorization reports
if (OPT_VEC_REPORT)
    if (MSVC)
        add_compile_options(
            /Qvec-report:1
            )
    elseif(CMAKE_COMPILER_IS_GNUCXX)
        add_compile_options(
            -fopt-info-vec-all
            )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Rpass=loop-vectorize
            )
    endif()
endif()

if (OPT_FAST_MATH)
    if (MSVC)
        add_compile_options(
            /fp:fast
            )
    elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -ffast-math
            )
    endif()
endif()

# ThreadSanitizer
if (OPT_TSAN)
    if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
        add_link_options(-fsanitize=thread)
    else()
        message("thread sanitizer not supported for this compiler")
    endif()
endif()

# AddressSanitizer
if (OPT_ASAN)
    if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    else()
        message("address sanitizer not supported for this compiler")
    endif()
endif()

# Set optimization flags
if (MSVC)
    add_compile_options(
        /W3 /arch:AVX /Zc:sizedDealloc-
        )
elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic -march=native
        )
endif()

# Project
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory("lmv")
add_subdirectory("jacobi")
add_subdirectory("seidel")

# Tests
include(CTest)
enable_testing()

add_test(NAME jacobi COMMAND jacobi_test)
add_test(NAME lmv COMMAND lmv_test)
add_test(NAME seidel COMMAND seidel_test)
