#include "losses/MeanSquaredError.hpp"
#include "optimizers/GradientDescent.hpp"
#include "gtest/gtest.h"

TEST(GradientDescentTest, validate_gradient_descent)
{
    constexpr std::size_t Examples = 7;
    constexpr std::size_t Features = 1;
    using GradientFunctionTest = ml_tools::MeanSquaredError::GradientFunction<double, Examples, Features>;
    using CostFunctionTest = ml_tools::MeanSquaredError::CostFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    -2.5, 
                                                     -1.0,  
                                                      0.1,  
                                                      1.9,
                                                      2.7,
                                                      3.5,
                                                      4.3});

    Eigen::Matrix<double, Examples, 1>        y({    6.7, 
                                                     4.2, 
                                                     4.9, 
                                                     3.84,
                                                     3.29,
                                                     2.81,
                                                     2.33});

    GradientFunctionTest gradientFunctionTest(x, y);
    CostFunctionTest costFunctionTest(x, y);

    ml_tools::ModelParameters<double, Features> resultExpected;
    ml_tools::ModelParameters<double, Features> initialParameters;

    resultExpected.weights << -0.549633;
    resultExpected.bias << 4.71667;

    initialParameters.weights << 0;
    initialParameters.bias << 0;

    ml_tools::GradientDescent<double, GradientFunctionTest, CostFunctionTest, Features> gd(gradientFunctionTest, costFunctionTest, 1500, 150, 0.05, 0);

    auto onEpoch = [](std::size_t, double, ml_tools::ModelParameters<double, Features>&) { };

    auto onDone = [&resultExpected](ml_tools::ModelParameters<double, Features>& result)
    {
        EXPECT_TRUE(result.IsApprox(resultExpected));
    };

    gd.Minimize(initialParameters, onEpoch, onDone);
}