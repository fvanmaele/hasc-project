add_library(jacobi INTERFACE jacobi.h)
target_link_libraries(jacobi INTERFACE
    OpenMP::OpenMP_CXX)

add_executable(jacobi_main main.cpp)
target_link_libraries(jacobi_main PRIVATE
    jacobi utility)

add_executable(jacobi_test test.cpp)
target_link_libraries(jacobi_test PRIVATE
    jacobi utility)
add_test(jacobi_test jacobi_test)

add_executable(jacobi_benchmark benchmark.cpp)
target_link_libraries(jacobi_benchmark PRIVATE
    jacobi utility benchmark::benchmark benchmark::benchmark_main)
