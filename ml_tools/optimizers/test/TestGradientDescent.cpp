#include "optimizers/GradientDescent.hpp"
#include "gtest/gtest.h"

namespace
{
    template<typename T, std::size_t NumberOfExamples, std::size_t NumberOfFeatures>
    struct CostFunctionImpl
    {
        CostFunctionImpl(ml_tools::Matrix<double, Examples, Features>& _x, ml_tools::Matrix<double, Examples, Features>& _y)
            : x(_x)
            , y(_y)
        {}

        T operator()(ml_tools::ModelParametersRef<T, NumberOfFeatures> parameters, T regularization)
        {
#if 0
            T m = parameters.w.dimension.rows;
            T cost = 0;
            T& w = parameters.w;
            T& b = parameters.b;

            for (std::size_t i = 0; i < m; i++)
            {
                auto temp = ((w * x(i, 0) + b) - y(i, 0))
                cost += temp * temp;
            }

            return cost / (2 * m);
#else
            return 0;
#endif
        }

        ml_tools::Matrix<double, Examples, Features>& x;
        ml_tools::Matrix<double, Examples, Features>& y;
    };

    template<typename T, std::size_t NumberOfExamples, std::size_t NumberOfFeatures>
    struct GradientFunctionImpl
    {
        GradientFunctionImpl(ml_tools::Matrix<double, Examples, Features>& _x, ml_tools::Matrix<double, Examples, Features>& _y)
            : x(_x)
            , y(_y)
        {}

        ml_tools::ModelParametersRef<T, NumberOfFeatures> operator()(ml_tools::ModelParametersRef<T, NumberOfFeatures> parameters, T regularization)
        {
#if 0
            T m = parameters.w.dimension.rows;
            T cost = 0;
            T& dj_dw = gradientCostFunction.w;
            T& dj_db = gradientCostFunction.b;

            dj_dw = 0;
            dj_db = 0;

            for (std::size_t i = 0; i < m; i++)
            {
                dj_dw += ((w * x(i, 0) + b) - y(i, 0)) * x(i, 0);
                dj_db += ((w * x(i, 0) + b) - y(i, 0));
            }

            dj_dw /= m;
            dj_db /= m;
#endif
            return gradientCostFunction;
        }

        ml_tools::ModelParameters<T, NumberOfFeatures> gradientCostFunction;
        ml_tools::Matrix<double, Examples, Features>& x;
        ml_tools::Matrix<double, Examples, Features>& y;
    };
}

TEST(GradientDescentTest, test)
{
    constexpr std::size_t Examples = 4;
    constexpr std::size_t Features = 1;
    using GradientFunctionTest = GradientFunctionImpl<double, Examples, Features>;
    using CostFunctionTest = CostFunctionImpl<double, Examples, Features>;

    ml_tools::Matrix<double, Examples, Features> x({2.0, 4.0,  6.0,  8.0});
    ml_tools::Matrix<double, Examples, 1>        y({4.5, 8.5, 12.5, 16.5});

    GradientFunctionTest gradientFunctionTest(x, y);
    CostFunctionTest costFunctionTest(x, y);

    ml_tools::GradientDescent<double, GradientFunctionTest, CostFunctionTest, Features> gd(gradientFunctionTest, costFunctionTest, 1, 1, 0.001, 0);

}