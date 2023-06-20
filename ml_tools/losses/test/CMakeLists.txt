add_executable(ml_tools.losses_test)
emil_build_for(ml_tools.losses_test)
emil_add_test(ml_tools.losses_test)

target_link_libraries(ml_tools.losses_test PUBLIC
    gmock_main
    ml_tools.losses
)

target_sources(ml_tools.losses_test PRIVATE
    TestMeanSquaredError.cpp
)
