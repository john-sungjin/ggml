#
# wabisabi

set(TEST_TARGET wabisabi)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
