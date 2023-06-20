#ifndef MLTOOLS_MODELS_MODEL_HPP
#define MLTOOLS_MODELS_MODEL_HPP

#include <Eigen/Geometry>
#include "infra/util/Function.hpp"
#include <functional>
#include <cmath>

namespace ml_tools
{
    template<typename T, std::size_t NumberOfFeatures>
    struct ModelParameters
    {
        Eigen::Matrix<T, NumberOfFeatures, 1> weights;
        Eigen::Matrix<T, 1, 1> bias;

        bool IsApprox(const ModelParameters<T, NumberOfFeatures>& rhs)
        {
            return rhs.bias.isApprox(this->bias, 1e-4) && rhs.weights.isApprox(this->weights, 1e-4);
        }

        ModelParameters<T, NumberOfFeatures>& operator-=(const ModelParameters<T, NumberOfFeatures>& x)
        {
            this->weights -= x.weights;
            this->bias -= x.bias;

            return *this;
        }

        ModelParameters<T, NumberOfFeatures>& operator*(T scalar)
        {
            this->weights * scalar;
            this->bias * scalar;

            return *this;
        }
    };
}

#endif
