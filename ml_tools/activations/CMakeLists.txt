add_library(ml_tools.activations ${EMIL_EXCLUDE_FROM_ALL} STATIC)

target_include_directories(ml_tools.activations PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(ml_tools.activations
    infra.util
    ml_tools.linear_algebra
    ml_tools.models
)

target_sources(ml_tools.activations PRIVATE
    Relu.hpp
    Relu.cpp
    Sigmoid.hpp
    Sigmoid.cpp
    SoftMax.hpp
    SoftMax.cpp
)
