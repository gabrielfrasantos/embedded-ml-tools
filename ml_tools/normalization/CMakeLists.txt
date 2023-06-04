add_library(ml_tools.normalization ${EMIL_EXCLUDE_FROM_ALL} STATIC)

target_include_directories(ml_tools.normalization PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(ml_tools.normalization
    infra.util
    ml_tools.linear_algebra
)

target_sources(ml_tools.normalization PRIVATE
    ZScore.cpp
    ZScore.hpp
)
