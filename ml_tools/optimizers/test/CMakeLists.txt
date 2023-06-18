add_executable(ml_tools.optimizers_test)
emil_build_for(ml_tools.optimizers_test)
emil_add_test(ml_tools.optimizers_test)

target_link_libraries(ml_tools.optimizers_test PUBLIC
    gmock_main
    ml_tools.losses
    ml_tools.optimizers
)

target_sources(ml_tools.optimizers_test PRIVATE
    TestGradientDescent.cpp
)
