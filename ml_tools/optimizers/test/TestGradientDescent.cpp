#include "optimizers/GradientDescent.hpp"
#include "gtest/gtest.h"

namespace
{
    // Move the structs below to "losses/MeanSquaredError"

    template<typename T, std::size_t Examples, std::size_t Features>
    struct CostFunctionImpl
    {
        CostFunctionImpl(Eigen::Matrix<double, Examples, Features>& x, Eigen::Matrix<double, Examples, Features>& y)
            : input(x)
            , output(y)
        {}

        T operator()(ml_tools::ModelParameters<T, Features>& parameters, T regularization)
        {
            T cost = 0;

            auto& weights = parameters.weights;
            auto& bias = parameters.bias;
            auto examples = parameters.weights.rows();
            
            for (std::size_t i = 0; i < examples; i++)
            {
                auto temp = (weights * input(i, Eigen::all) + bias) - output(i, Eigen::all);
                cost += temp * temp;
            }

            return cost / (2 * examples);
        }

        Eigen::Matrix<double, Examples, Features>& input;
        Eigen::Matrix<double, Examples, 1>& output;
    };

    template<typename T, std::size_t Examples, std::size_t Features>
    struct GradientFunctionImpl
    {
        GradientFunctionImpl(Eigen::Matrix<double, Examples, Features>& x, Eigen::Matrix<double, Examples, 1>& y)
            : input(x)
            , output(y)
        {}

        ml_tools::ModelParameters<T, Features>& operator()(ml_tools::ModelParameters<T, Features>& parameters, T regularization)
        {

            auto examples = parameters.weights.rows();
            auto& dj_dw = gradientCostFunction.weights;
            auto& dj_db = gradientCostFunction.bias;
            auto& weights = parameters.weights;
            auto& bias = parameters.bias;

            dj_dw.setZero();
            dj_db.setZero();

            for (std::size_t i = 0; i < examples; i++)
            {
                dj_dw += ((weights * input(i, Eigen::all) + bias) - output(i, Eigen::all)) * input(i, Eigen::all);
                dj_db += ((weights * input(i, Eigen::all) + bias) - output(i, Eigen::all));
            }

            dj_dw /= examples;
            dj_db /= examples;

            return gradientCostFunction;
        }

        ml_tools::ModelParameters<T, Features> gradientCostFunction;
        Eigen::Matrix<double, Examples, Features>& input;
        Eigen::Matrix<double, Examples, 1>& output;
    };
}

TEST(GradientDescentTest, validate_gradient_descent)
{
    constexpr std::size_t Examples = 4;
    constexpr std::size_t Features = 1;
    using GradientFunctionTest = GradientFunctionImpl<double, Examples, Features>;
    using CostFunctionTest = CostFunctionImpl<double, Examples, Features>;

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