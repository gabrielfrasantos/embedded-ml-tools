add_library(ml_tools.models ${EMIL_EXCLUDE_FROM_ALL} STATIC)

target_include_directories(ml_tools.models PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(ml_tools.models
    infra.util
    ml_tools.linear_algebra
    Eigen3::Eigen
)

target_sources(ml_tools.models PRIVATE
    Model.cpp
    Model.hpp
)
