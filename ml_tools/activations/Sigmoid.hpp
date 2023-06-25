#ifndef MLTOOLS_ACTIVATIONS_SIGMOID_HPP
#define MLTOOLS_ACTIVATIONS_SIGMOID_HPP

#include <Eigen/Geometry>
#include <functional>
#include <cmath>
#include "models/Model.hpp"

namespace ml_tools
{
    struct Sigmoid
    {
        template<typename T, std::size_t Rows, std::size_t Columns>
        static Eigen::Matrix<T, Rows, Columns> f(Eigen::Matrix<T, Rows, Columns> z)
        {
            auto sigmoid = [](T x) -> T
            {
                return 1 / (1 + std::exp(-x));
            }

            z.unaryExpr(sigmoid);

            return z;
        }

        template<typename T, std::size_t Rows, std::size_t Columns>
        static Eigen::Matrix<T, Rows, Columns> df(Eigen::Matrix<T, Rows, Columns> z)
        {
            auto h = f(z);

            return h * (1 - h);
        }

        template<typename T>
        static T f(T z)
        {
            return 1 / (1 + std::exp(-z));
        }

        template<typename T>
        static T df(T z)
        {
            auto h = f(z);

            return h * (1 - h);
        }
    };
}

#endif
