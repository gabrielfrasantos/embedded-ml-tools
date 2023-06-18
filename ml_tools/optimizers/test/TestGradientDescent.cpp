#include "losses/MeanSquaredError.hpp"
#include "optimizers/GradientDescent.hpp"
#include "gtest/gtest.h"

TEST(GradientDescentTest, validate_gradient_descent)
{
    constexpr std::size_t Examples = 4;
    constexpr std::size_t Features = 1;
    using GradientFunctionTest = ml_tools::MeanSquaredError::GradientFunction<double, Examples, Features>;
    using CostFunctionTest = ml_tools::MeanSquaredError::CostFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    2.0, 
                                                     4.0,  
                                                     6.0,  
                                                     8.0});

    Eigen::Matrix<double, Examples, 1>        y({    4.5, 
                                                     8.5, 
                                                    12.5, 
                                                    16.5});

    GradientFunctionTest gradientFunctionTest(x, y);
    CostFunctionTest costFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;

    initialParameters.weights << 2;
    initialParameters.bias << 0.5;

    ml_tools::GradientDescent<double, GradientFunctionTest, CostFunctionTest, Features> gd(gradientFunctionTest, costFunctionTest, 1, 1, 0.001, 0);

    gd.Minimize(initialParameters, [](std::size_t epoch, double loss, ml_tools::ModelParameters<double, Features>& grandient){}, [](ml_tools::ModelParameters<double, Features>&){});
}