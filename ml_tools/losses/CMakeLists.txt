add_library(ml_tools.losses ${EMIL_EXCLUDE_FROM_ALL} STATIC)

target_include_directories(ml_tools.losses PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(ml_tools.losses
    infra.util
    ml_tools.linear_algebra
    ml_tools.models
    Eigen3::Eigen
)

target_sources(ml_tools.losses PRIVATE
    BinaryCrossEntropy.cpp
    BinaryCrossEntropy.hpp
    MeanSquaredError.cpp
    MeanSquaredError.hpp
)

add_subdirectory(test)
