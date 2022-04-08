cmake_minimum_required(VERSION 3.12)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)"
        FORCE
  )
endif()
project(matrix_multiplication_parallel)

# HPX
find_package(HPX REQUIRED)

set (program par_matrix_mult)
add_hpx_executable(${program} ESSENTIAL SOURCES ${program}.cpp)