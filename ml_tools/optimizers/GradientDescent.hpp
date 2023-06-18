#ifndef MLTOOLS_OPTIMIZERS_GRADIENT_DESCENT_HPP
#define MLTOOLS_OPTIMIZERS_GRADIENT_DESCENT_HPP

#include <Eigen/Geometry>
#include "infra/util/Function.hpp"
#include <functional>
#include <cmath>

namespace ml_tools
{
    template<typename T>
    constexpr T ceiling_for_division(T dividend, T divisor)
    {
        return (divisor > 0 && dividend > 0) ? ((dividend + divisor - 1) / divisor) : 0;
    }

    template<typename T, std::size_t NumberOfFeatures>
    struct ModelParameters
    {
        Eigen::Matrix<T, NumberOfFeatures, 1> weights;
        Eigen::Matrix<T, 1, 1> bias;

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

    template<typename T, class GradientFunction, class CostFunction, std::size_t NumberOfFeatures>
    class GradientDescent
    {
    public:
        using EpochCallback = std::function<void(std::size_t epoch, T loss, ModelParameters<T, NumberOfFeatures>&)>;
        using OnDoneCallback = std::function<void(ModelParameters<T, NumberOfFeatures>&)>;

        explicit GradientDescent(GradientFunction& gradientFunction, CostFunction& costFunction, std::size_t iterations, std::size_t epoch, T learningRate, T regularization = 0);

        void Minimize(ModelParameters<T, NumberOfFeatures>& initialParameters, const EpochCallback& onEpochCompleted, const OnDoneCallback& onDone);

    private:
        GradientFunction& gradientFunction;
        CostFunction& costFunction;
        std::size_t iterations;
        std::size_t epoch;
        T learningRate;
        T regularization;

        ModelParameters<T, NumberOfFeatures> parameters;
    };

    //// Implementation ////

    template<typename T, class GradientFunction, class CostFunction, std::size_t NumberOfFeatures>
    GradientDescent<T, GradientFunction, CostFunction, NumberOfFeatures>::GradientDescent(GradientFunction& gradientFunction, CostFunction& costFunction, std::size_t iterations, std::size_t epoch, T learningRate, T regularization)
        : gradientFunction(gradientFunction)
        , costFunction(costFunction)
        , iterations(iterations)
        , epoch(epoch)
        , learningRate(learningRate)
        , regularization(regularization)
    {}

    template<typename T, class GradientFunction, class CostFunction, std::size_t NumberOfFeatures>
    void GradientDescent<T, GradientFunction, CostFunction, NumberOfFeatures>::Minimize(ModelParameters<T, NumberOfFeatures>& initialParameters, const EpochCallback& onEpochCompleted, const OnDoneCallback& onDone)
    {
        std::size_t currentEpoch = 0;

        parameters = initialParameters;

        for (std::size_t i = 0; i < iterations; ++i)
        {
            auto gradient = gradientFunction(parameters, regularization);

            parameters -= gradient * learningRate;

            auto cost = costFunction(parameters, regularization);

            if (i % ceiling_for_division(iterations, epoch) == 0)
            {
                onEpochCompleted(currentEpoch, cost, gradient);

                currentEpoch++;
            }
        }

        onDone(parameters);
    }
}

#endif
