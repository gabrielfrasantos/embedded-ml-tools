add_library(ml_tools.linear_algebra ${EMIL_EXCLUDE_FROM_ALL} STATIC)

target_include_directories(ml_tools.linear_algebra PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(ml_tools.linear_algebra
    infra.util
)

target_sources(ml_tools.linear_algebra PRIVATE
    Matrix.hpp
    Matrix.cpp
    Mean.hpp
    Mean.cpp
    StandardDeviation.hpp
    StandardDeviation.cpp
)

add_subdirectory(test)