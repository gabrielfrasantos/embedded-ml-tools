#ifndef MLTOOLS_OPTIMIZERS_GRADIENT_DESCENT_HPP
#define MLTOOLS_OPTIMIZERS_GRADIENT_DESCENT_HPP

#include "infra/util/Function.hpp"
#include "linear_algebra/Matrix.hpp"
#include <cmath>

namespace ml_tools
{
    template<typename T, std::size_t NumberOfFeatures>
    struct ModelParameters
    {
        Matrix<T, NumberOfFeatures, 1> w;
        Matrix<T, 1, 1> b;
    };

    template<typename T, std::size_t NumberOfFeatures>
    struct ModelParametersRef
    {
        Matrix<T, NumberOfFeatures, 1>& w;
        Matrix<T, 1, 1>& b;
    };

    template<typename T, class GradientFunction, class CostFunction, std::size_t NumberOfFeatures>
    class GradientDescent
    {
    public:
        using EpochCallback = infra::Function<void(std::size_t epoch, T loss)>;
        using OnDoneCallback = infra::Function<void(ModelParametersRef<T, NumberOfFeatures>)>;

        explicit GradientDescent(GradientFunction& gradientFunction, CostFunction& costFunction, std::size_t iterations, std::size_t epoch, T learningRate, T regularization = 0);

        void minimize(ModelParametersRef<T, NumberOfFeatures> initialParameters, const EpochCallback& onEpochCompleted, const OnDoneCallback& onDone);

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
    void GradientDescent<T, GradientFunction, CostFunction, NumberOfFeatures>::minimize(ModelParametersRef<T, NumberOfFeatures> initialParameters, const EpochCallback& onEpochCompleted, const OnDoneCallback& onDone)
    {
        std::size_t currentEpoch = 0;

        parameters = initialParameters;

        for (std::size_t i = 0; i < iterations; ++i)
        {
            auto gradient = gradientFunction(parameters, regularization)

            parameters -= learningRate * gradient;

            auto cost = costFunction(parameters, regularization);

            if (i % ceilf(static_cast<float>(iterations) / static_cast<float>(epoch)) == 0)
            {
                onEpochCompleted(currentEpoch, cost, gradient);

                currentEpoch++;
            }
        }

        onDone(parameters);
    }
}

#endif
