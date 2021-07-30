add_library(seidel INTERFACE)
target_link_libraries(seidel INTERFACE
    OpenMP::OpenMP_CXX)
target_sources(seidel INTERFACE 
    seidel.h)

add_executable(seidel_main main.cpp)
target_link_libraries(seidel_main PRIVATE
    seidel utility)

add_executable(seidel_test test.cpp)
target_link_libraries(seidel_test PRIVATE
    seidel utility)

add_executable(seidel_benchmark benchmark.cpp)
target_link_libraries(seidel_benchmark PRIVATE
    seidel utility benchmark::benchmark benchmark::benchmark_main)
