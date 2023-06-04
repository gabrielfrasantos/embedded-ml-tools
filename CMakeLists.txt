cmake_minimum_required(VERSION 3.21)

project(embedded-ml-tools VERSION 1.0.0)

include(FetchContent)

FetchContent_Declare(
    emil
    GIT_REPOSITORY https://github.com/gabrielfrasantos/embedded-infra-lib.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(emil)

add_subdirectory(ml_tools)

add_executable(ml-tools Main.cpp)

target_link_libraries(ml-tools PUBLIC
    infra.event
    ml_tools.activations
    ml_tools.linear_algebra
    ml_tools.normalization
    ml_tools.optimizers
)

