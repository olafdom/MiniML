cmake_minimum_required(VERSION 3.15...3.27)

project(miniml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    ERROR_VARIABLE PYBIND11_ERROR
    RESULT_VARIABLE PYBIND11_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYBIND11_STATUS EQUAL 0)
    message(FATAL_ERROR "pybind11 not found: ${PYBIND11_ERROR}")
endif()

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core 
    src/knn.cpp 
    src/bindings.cpp
    src/utils/metrics.cpp
)

install(TARGETS _core DESTINATION miniml)

if(MINGW)
    target_link_options(_core PRIVATE -static)
endif()
