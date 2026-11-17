cmake_minimum_required(VERSION 3.16)
project(PI_NSGA2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimización: Release para benchmarks, Debug con sanitizers
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenMP)

# ─── Includes ─────────────────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ─── Fuentes ──────────────────────────────────────────────────────────────────
set(CORE_SOURCES
    src/tree_node.cpp
    src/pde_problems.cpp
    src/numerical_solver.cpp
    src/pi_solver.cpp
    src/optimizer_nelder_mead.cpp
    src/optimizer_de.cpp
    src/oracle.cpp
)
add_library(pisr_core STATIC ${CORE_SOURCES})
target_link_libraries(pisr_core m stdc++fs -fopenmp)

# ─── Ejecutable principal ─────────────────────────────────────────────────────
add_executable(PISR-NSGA-II main.cpp)
target_link_libraries(PISR-NSGA-II pisr_core)

# ─── Tests (correr antes de cada experimento: ./run_tests.sh) ────────────────
add_executable(oracle_test tests/oracle_test.cpp)
target_link_libraries(oracle_test pisr_core)
add_executable(gradient_test tests/gradient_test.cpp)
target_link_libraries(gradient_test pisr_core)

# ─── Información de compilación ───────────────────────────────────────────────
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags  : ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
