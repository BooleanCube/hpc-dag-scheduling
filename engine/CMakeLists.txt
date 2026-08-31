cmake_minimum_required(VERSION 3.22)
project(engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(engine src/main.cpp)
target_link_libraries(engine PRIVATE MPI::MPI_CXX)

# hpcctl deploy rsyncs the build tree to /shared/engine/, and hpcctl submit runs
# ${HPCCTL_ENGINE_BINARY}, which defaults to /shared/engine/bin/engine. bin/ here is
# what makes those defaults line up.
set_target_properties(engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
