add_definitions(-std=c++11 -msse4)

if (UBITS_AVX2)
    add_definitions(-DUBITS_AVX2 -mavx2)
endif()

set(UBITS_BINUTILS_DIR ${UBITS_BINUTILS_DIR} CACHE PATH "path to binutils bin directory")
if (UBITS_BINUTILS_DIR)
    add_definitions(-B ${UBITS_BINUTILS_DIR})
endif()

include_directories(${PROJECT_SOURCE_DIR})
add_executable(bitset_benchmark bitset_benchmark.cpp)
add_executable(binsearch_benchmark binary_search.cpp)
add_executable(intersection_benchmark intersection.cpp)

install(TARGETS
        bitset_benchmark
        binsearch_benchmark
        intersection_benchmark
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Sometimes I need to compile the benchmarks with different version of g++ and run them on different machines,
# and it is convenient to distribute libstdc++.so with them.
execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-file-name libstdc++.so.6 OUTPUT_VARIABLE LIBSTDCXX_SO)
string(REGEX REPLACE "\n\n*" "" LIBSTDCXX_SO "${LIBSTDCXX_SO}")
# Usually libstdc++.so.6 is a symlink, though I want to copy the actual file.
# The CMake command install(FILES ...) or file(copy ...) does not copy the actual file, so I have to use "cp" .
get_filename_component(LIBSTDCXX_SO_FILENAME ${LIBSTDCXX_SO} NAME)
install(CODE "execute_process(COMMAND cp ${LIBSTDCXX_SO} ${CMAKE_INSTALL_PREFIX}/bin/${LIBSTDCXX_SO_FILENAME})")
