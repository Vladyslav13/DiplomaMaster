cmake_minimum_required(VERSION 3.12)

project(MasterDiplomaVlad)

set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/assets)

add_compile_definitions(ASSETS_DIR="${ASSETS_DIR}")

if (MULTIPROCESSOR_COMPILATION)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

add_subdirectory(modules/recognitionlib)
add_subdirectory(modules/sandbox)