cmake_minimum_required(VERSION 3.21)

project(embedded-ml-tools VERSION 1.0.0)

include(FetchContent)

FetchContent_Declare(
  emil
  GIT_REPOSITORY    https://github.com/gabrielfrasantos/embedded-infra-lib.git
  GIT_TAG           main
)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
  GIT_TAG           3.4.0
  GIT_SHALLOW       TRUE)

FetchContent_Declare(
  matplotplusplus
  GIT_REPOSITORY    https://github.com/alandefreitas/matplotplusplus.git
  GIT_TAG           v1.2.0)

set(EIGEN_BUILD_DOC           OFF)
set(EIGEN_BUILD_PKGCONFIG     OFF)
set(BUILD_TESTING             OFF)
set(MATPLOTPP_BUILD_EXAMPLES  OFF)
set(MATPLOTPP_BUILD_TESTS     OFF)

FetchContent_MakeAvailable(emil eigen matplotplusplus)

add_subdirectory(ml_tools)

add_executable(ml-tools Main.cpp)

target_link_libraries(ml-tools PUBLIC
    infra.event
    ml_tools.activations
    ml_tools.linear_algebra
    ml_tools.normalization
    ml_tools.optimizers
)


