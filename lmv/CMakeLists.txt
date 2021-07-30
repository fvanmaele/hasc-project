add_executable(lmv_main main.cpp)

add_executable(lmv_test test.cpp)

add_executable(lmv_benchmark benchmark.cpp)
target_link_libraries(lmv_benchmark PRIVATE
    benchmark::benchmark benchmark::benchmark_main)

#add_executable(lmv_benchmark_blocked benchmark_blocked.cpp)
#target_link_libraries(lmv_benchmark_blocked PRIVATE
#    lmv utility benchmark::benchmark benchmark::benchmark_main)
