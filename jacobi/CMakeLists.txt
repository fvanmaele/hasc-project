add_executable(jacobi_main main.cpp)
target_link_libraries(jacobi_main PRIVATE
    OpenMP::OpenMP_CXX)

add_executable(jacobi_test test.cpp)
target_link_libraries(jacobi_test PRIVATE
    OpenMP::OpenMP_CXX)

add_executable(jacobi_benchmark benchmark.cpp)
target_link_libraries(jacobi_benchmark PRIVATE
    OpenMP::OpenMP_CXX benchmark::benchmark benchmark::benchmark_main)
