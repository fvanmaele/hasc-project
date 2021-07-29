add_library(lmv INTERFACE lmv_seq.h lmv_vcl.h)
#target_link_libraries(lmv INTERFACE
#    vectorclass::vectorclass)

add_executable(lmv_main main.cpp)
target_link_libraries(lmv_main PRIVATE
    lmv utility)

add_executable(lmv_test test.cpp)
target_link_libraries(lmv_test PRIVATE
    lmv utility)

add_executable(lmv_benchmark benchmark.cpp)
target_link_libraries(lmv_benchmark PRIVATE
    lmv utility benchmark::benchmark benchmark::benchmark_main)

#add_executable(lmv_benchmark_blocked benchmark_blocked.cpp)
#target_link_libraries(lmv_benchmark_blocked PRIVATE
#    lmv utility benchmark::benchmark benchmark::benchmark_main)
