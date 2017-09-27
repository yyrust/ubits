cmake_minimum_required(VERSION 2.8)
project(ubits)

option(UBITS_AVX2 "Enable algorithms using AVX2 instruction set" OFF)

add_subdirectory(ubits)
add_subdirectory(benchmark)
