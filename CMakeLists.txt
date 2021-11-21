cmake_minimum_required(VERSION 3.20)
project(pv021_project)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(pv021_project
        main.cpp
        main.h
        Net.cpp
#        NetThread.cpp
        Net.h
        Csv.cpp
        Csv.h
#        Operations.hpp
        OperationsThreads.hpp preprocessors/Scaler/Scaler.cpp preprocessors/Scaler/Scaler.h initializers/Glorot/glorot_uniform.cpp initializers/Glorot/glorot_uniform.h initializers/Kaiming/kaiming.cpp initializers/Kaiming/kaiming.h initializers/Xavier/xavier.cpp initializers/Xavier/xavier.h optimizers/Adam/adam.cpp optimizers/Adam/adam.h)
