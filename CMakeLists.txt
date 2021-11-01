cmake_minimum_required(VERSION 3.20)
project(pv021_project)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(pv021_project
        main.cpp
        main.h
        Net.cpp
        Net.h
        Csv.cpp
        Csv.h
        Operations.hpp)
