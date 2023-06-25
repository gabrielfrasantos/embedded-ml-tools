#ifndef MLTOOLS_LOSSES_CROSS_ENTROPY_HPP
#define MLTOOLS_LOSSES_CROSS_ENTROPY_HPP

#include <Eigen/Geometry>
#include <functional>
#include <cmath>
#include "activations/Sigmoid.hpp"

namespace ml_tools
{
    namespace MeanSquaredError
    {
        namespace
        {
            template<typename T, std::size_t Examples, std::size_t Features>
            T AccumulativeSumProductBetweenWeightsAndInput(std::size_t example, ModelParameters<T, Features> parameters, Eigen::Matrix<T, Examples, Features>& input)
            {
                auto& weights = parameters.weights;
                auto& bias = parameters.bias;
                auto features = input.cols();

                T z = 0;

                for (std::size_t j = 0; j < features; j++)
                    z += weights(example, 0) * input(example, j);

                return z + bias(0, 0);
            }
        }

        template<typename T, std::size_t Examples, std::size_t Features>
        struct CostFunction
        {
            CostFunction(Eigen::Matrix<T, Examples, Features>& x, Eigen::Matrix<T, Examples, Features>& y)
                : input(x)
                , output(y)
            {}

            T operator()(ml_tools::ModelParameters<T, Features>& parameters, T regularization)
            {
                T cost = 0;

                auto examples = input.rows();
                
                for (std::size_t i = 0; i < examples; i++)
                {
                    auto z = AccumulativeSumProductBetweenWeightsAndInput(i, parameters, input);

                    cost += -y(i, 0) * std::log(Sigmoid::f(z)) - (1 - y(i, 0)) * std::log(1 - Sigmoid::f(z));
                }

                return cost / examples;
            }

            Eigen::Matrix<T, Examples, Features>& input;
            Eigen::Matrix<T, Examples, 1>& output;
        };

        template<typename T, std::size_t Examples, std::size_t Features>
        struct GradientFunction
        {
            GradientFunction(Eigen::Matrix<T, Examples, Features>& x, Eigen::Matrix<T, Examples, 1>& y)
                : input(x)
                , output(y)
            {}

            ml_tools::ModelParameters<T, Features>& operator()(ml_tools::ModelParameters<T, Features>& parameters, T regularization)
            {
                /*auto examples = input.rows();
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

                return gradientCostFunction;*/

                auto examples = input.rows();
                auto& dj_dw = gradientCostFunction.weights;
                auto& dj_db = gradientCostFunction.bias;

                dj_dw.setZero();
                dj_db.setZero();
                
                for (std::size_t i = 0; i < examples; i++)
                {
                    auto f_wb = Sigmoid::f(AccumulativeSumProductBetweenWeightsAndInput(i, parameters, input));

                    cost += -y(i, 0) * std::log(Sigmoid::f(z)) - (1 - y(i, 0)) * std::log(1 - Sigmoid::f(z));
                }

                return cost / examples;


            }

            ml_tools::ModelParameters<T, Features> gradientCostFunction;
            Eigen::Matrix<T, Examples, Features>& input;
            Eigen::Matrix<T, Examples, 1>& output;
        };
    }
}

#endif
