cmake_minimum_required(VERSION 3.15...3.27)

project(miniml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
