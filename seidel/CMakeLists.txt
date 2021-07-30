add_executable(seidel_main main.cpp)
target_link_libraries(seidel_main PRIVATE
    OpenMP::OpenMP_CXX)

add_executable(seidel_test test.cpp)
target_link_libraries(seidel_test PRIVATE
    OpenMP::OpenMP_CXX)

add_executable(seidel_benchmark benchmark.cpp)
target_link_libraries(seidel_benchmark PRIVATE
    OpenMP::OpenMP_CXX benchmark::benchmark benchmark::benchmark_main)
