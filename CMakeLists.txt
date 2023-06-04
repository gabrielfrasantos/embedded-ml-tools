cmake_minimum_required(VERSION 3.21)

project(embedded-ml-tools VERSION 1.0.0)

include(FetchContent)

FetchContent_Declare(
    emil
    GIT_REPOSITORY https://github.com/gabrielfrasantos/embedded-infra-lib.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(emil)


