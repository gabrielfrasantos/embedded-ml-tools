#ifndef MLTOOLS_LOSSES_MEAN_SQUARED_ERROR_HPP
#define MLTOOLS_LOSSES_MEAN_SQUARED_ERROR_HPP

#include <Eigen/Geometry>
#include <functional>
#include <cmath>
#include "models/Model.hpp"

namespace ml_tools
{
    namespace MeanSquaredError
    {
        template<typename T, std::size_t Examples, std::size_t Features>
        struct CostFunction
        {
            CostFunction(Eigen::Matrix<double, Examples, Features>& x, Eigen::Matrix<double, Examples, Features>& y)
                : input(x)
                , output(y)
            {}

            T operator()(ml_tools::ModelParameters<T, Features>& parameters, T regularization)
            {
                T cost = 0;

                auto& weights = parameters.weights;
                auto& bias = parameters.bias;
                auto examples = input.rows();
                
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
        struct GradientFunction
        {
            GradientFunction(Eigen::Matrix<double, Examples, Features>& x, Eigen::Matrix<double, Examples, 1>& y)
                : input(x)
                , output(y)
            {}

            ml_tools::ModelParameters<T, Features>& operator()(ml_tools::ModelParameters<T, Features>& parameters, T regularization)
            {

                auto examples = input.rows();
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
}

#endif
