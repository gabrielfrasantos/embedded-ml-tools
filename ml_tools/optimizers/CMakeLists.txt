add_library(ml_tools.optimizers ${EMIL_EXCLUDE_FROM_ALL} STATIC)

target_include_directories(ml_tools.optimizers PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(ml_tools.optimizers
    infra.util
    ml_tools.linear_algebra
    ml_tools.models
    Eigen3::Eigen
)

target_sources(ml_tools.optimizers PRIVATE
    GradientDescent.cpp
    GradientDescent.hpp
)

add_subdirectory(test)
