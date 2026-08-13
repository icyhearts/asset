cmake_minimum_required(VERSION 3.16)

project(cpp_guard_kgp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(kgp kgp.cpp)
add_executable(kgp_wl kgp_wl.cpp)

target_compile_options(kgp PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(kgp_wl PRIVATE -Wall -Wextra -Wpedantic)
