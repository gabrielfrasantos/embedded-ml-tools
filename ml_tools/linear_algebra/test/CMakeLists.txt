add_executable(ml_tools.linear_algebra_test)
emil_build_for(ml_tools.linear_algebra_test BOOL EMIL_BUILD_TESTS)
emil_add_test(ml_tools.linear_algebra_test)

target_link_libraries(ml_tools.linear_algebra_test PUBLIC
    gmock_main
    ml_tools.linear_algebra
)

target_sources(ml_tools.linear_algebra_test PRIVATE
    TestMatrix.cpp
)
