#include "losses/MeanSquaredError.hpp"
#include "optimizers/GradientDescent.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

MATCHER_P(IsMatrixEqual, reference, "Checking if matrix are the same") { return args.isApprox(reference, 1e-4); }

TEST(MeanSquaredErrorTest, validate_cost_function_1)
{
    constexpr std::size_t Examples = 4;
    constexpr std::size_t Features = 1;
    using CostFunctionTest = ml_tools::MeanSquaredError::CostFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    2.0, 
                                                     4.0,  
                                                     6.0,  
                                                     8.0});

    Eigen::Matrix<double, Examples, 1>        y({    7, 
                                                     11, 
                                                     15, 
                                                     19});

    CostFunctionTest costFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;

    initialParameters.weights << 2;
    initialParameters.bias << 3;

    EXPECT_DOUBLE_EQ(costFunctionTest(initialParameters, 0), 0);
}

TEST(MeanSquaredErrorTest, validate_cost_function_2)
{
    constexpr std::size_t Examples = 4;
    constexpr std::size_t Features = 1;
    using CostFunctionTest = ml_tools::MeanSquaredError::CostFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    2.0, 
                                                     4.0,  
                                                     6.0,  
                                                     8.0});

    Eigen::Matrix<double, Examples, 1>        y({    7, 
                                                     11, 
                                                     15, 
                                                     19});

    CostFunctionTest costFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;

    initialParameters.weights << 2;
    initialParameters.bias << 1;

    EXPECT_DOUBLE_EQ(costFunctionTest(initialParameters, 0), 2);
}

TEST(MeanSquaredErrorTest, validate_cost_function_3)
{
    constexpr std::size_t Examples = 5;
    constexpr std::size_t Features = 1;
    using CostFunctionTest = ml_tools::MeanSquaredError::CostFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    1.5, 
                                                     2.5,  
                                                     3.5,  
                                                     4.5,
                                                     1.5});

    Eigen::Matrix<double, Examples, 1>        y({    4, 
                                                     7, 
                                                     10, 
                                                     13,
                                                     5});

    CostFunctionTest costFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;

    initialParameters.weights << 1;
    initialParameters.bias << 0;

    EXPECT_DOUBLE_EQ(costFunctionTest(initialParameters, 0), 15.325);
}

TEST(MeanSquaredErrorTest, validate_cost_function_4)
{
    constexpr std::size_t Examples = 5;
    constexpr std::size_t Features = 1;
    using CostFunctionTest = ml_tools::MeanSquaredError::CostFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    1.5, 
                                                     2.5,  
                                                     3.5,  
                                                     4.5,
                                                     1.5});

    Eigen::Matrix<double, Examples, 1>        y({    4, 
                                                     7, 
                                                     10, 
                                                     13,
                                                     5});

    CostFunctionTest costFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;

    initialParameters.weights << 1;
    initialParameters.bias << 1;

    EXPECT_DOUBLE_EQ(costFunctionTest(initialParameters, 0), 10.725);
}

TEST(MeanSquaredErrorTest, validate_cost_function_5)
{
    constexpr std::size_t Examples = 5;
    constexpr std::size_t Features = 1;
    using CostFunctionTest = ml_tools::MeanSquaredError::CostFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    1.5, 
                                                     2.5,  
                                                     3.5,  
                                                     4.5,
                                                     1.5});

    Eigen::Matrix<double, Examples, 1>        y({    2, 
                                                     5, 
                                                     8, 
                                                     11,
                                                     3});

    CostFunctionTest costFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;

    initialParameters.weights << 1;
    initialParameters.bias << 1;

    EXPECT_DOUBLE_EQ(costFunctionTest(initialParameters, 0), 4.525);
}

TEST(MeanSquaredErrorTest, validate_gradient_function_1)
{
    constexpr std::size_t Examples = 4;
    constexpr std::size_t Features = 1;
    using GradientFunctionTest = ml_tools::MeanSquaredError::GradientFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    2.0, 
                                                     4.0,  
                                                     6.0,  
                                                     8.0});

    Eigen::Matrix<double, Examples, 1>        y({    4.5, 
                                                     8.5, 
                                                     12.5, 
                                                     16.5});

    GradientFunctionTest gradientFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;
    ml_tools::ModelParameters<double, Features> resultExpected;

    initialParameters.weights << 2;
    initialParameters.bias << 0.5;

    resultExpected.weights << 0;
    resultExpected.bias << 0;

    auto& gradient = gradientFunctionTest(initialParameters, 0);

    EXPECT_TRUE(gradient.IsApprox(resultExpected));
}

TEST(MeanSquaredErrorTest, validate_gradient_function_2)
{
    constexpr std::size_t Examples = 4;
    constexpr std::size_t Features = 1;
    using GradientFunctionTest = ml_tools::MeanSquaredError::GradientFunction<double, Examples, Features>;

    Eigen::Matrix<double, Examples, Features> x({    2.0, 
                                                     4.0,  
                                                     6.0,  
                                                     8.0});

    Eigen::Matrix<double, Examples, 1>        y({    6, 
                                                     9, 
                                                     12, 
                                                     15});

    GradientFunctionTest gradientFunctionTest(x, y);
    ml_tools::ModelParameters<double, Features> initialParameters;
    ml_tools::ModelParameters<double, Features> resultExpected;

    initialParameters.weights << 1.5;
    initialParameters.bias << 1;

    resultExpected.weights << -10;
    resultExpected.bias << -2;

    auto& gradient = gradientFunctionTest(initialParameters, 0);

    EXPECT_TRUE(gradient.IsApprox(resultExpected));
}