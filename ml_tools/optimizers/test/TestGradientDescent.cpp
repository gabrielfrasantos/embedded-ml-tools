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
    ml_tools::ModelParameters<double, Features> initialParameters;

    initialParameters.weights << -0.6;
    initialParameters.bias << 4.5;

    ml_tools::GradientDescent<double, GradientFunctionTest, CostFunctionTest, Features> gd(gradientFunctionTest, costFunctionTest, 25, 5, 0.01, 0);

    auto onEpoch = [](std::size_t epoch, double loss, ml_tools::ModelParameters<double, Features>& grandient)
    {
        std::cerr << "epoch: " << epoch << ", loss: " << loss << std::endl;
    };

    auto onDone = [](ml_tools::ModelParameters<double, Features>&)
    {

    };

    gd.Minimize(initialParameters, onEpoch, onDone);
}