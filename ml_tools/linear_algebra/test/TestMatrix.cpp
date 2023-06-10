#include "linear_algebra/Matrix.hpp"
#include "gtest/gtest.h"

TEST(MatrixTest, matrix_summation_i8)
{
    ml_tools::Matrix<int8_t, 3, 3> m1({
        { -3, -2, -1,
           0,  1,  2,
           3,  4,  5
    }});

    ml_tools::Matrix<int8_t, 3, 3> m2({
        {  3,  2,  1,
          -3,  1,  4,
          -7,  2, -6
    }});

    ml_tools::Matrix<int8_t, 3, 3> result({
        {  0,  0,  0,
          -3,  2,  6,
          -4,  6, -1
    }});

    m1 += m2;

    EXPECT_TRUE(m1 == result);
}

TEST(MatrixTest, matrix_subtraction_i8)
{
    ml_tools::Matrix<int8_t, 3, 3> m1({
        { -3, -2, -1,
           0,  1,  2,
           3,  4,  5
    }});

    ml_tools::Matrix<int8_t, 3, 3> m2({
        {  3,  2,  1,
          -3,  1,  4,
          -7,  2, -6
    }});

    ml_tools::Matrix<int8_t, 3, 3> result({
        { -6, -4, -2,
           3,  0, -2,
          10,  2, 11
    }});

    m1 -= m2;

    EXPECT_TRUE(m1 == result);
}

TEST(MatrixTest, scalar_multiplication_i8)
{
    ml_tools::Matrix<int8_t, 3, 3> m({
        { -3, -2, -1,
           0,  1,  2,
           3,  4,  5
    }});

    ml_tools::Matrix<int8_t, 3, 3> result({
        { -6, -4, -2,
           0,  2,  4,
           6,  8, 10
    }});

    m *= 2;

    EXPECT_TRUE(m == result);
}

TEST(MatrixTest, scalar_division_i8)
{
    ml_tools::Matrix<int8_t, 3, 3> m({
        { -6, -4, -2,
           0,  2,  4,
           6,  8, 10
    }});

    ml_tools::Matrix<int8_t, 3, 3> result({
        { -3, -2, -1,
           0,  1,  2,
           3,  4,  5
    }});

    m /= 2;

    EXPECT_TRUE(m == result);
}

TEST(MatrixTest, matrix_transpose_square_i8)
{
    ml_tools::Matrix<int8_t, 3, 3> m({
        { -6, -4, -2,
           0,  2,  4,
           6,  8, 10
    }});
    ml_tools::Matrix<int8_t, 3, 3> result;

    ml_tools::Matrix<int8_t, 3, 3> expectedResult({
        { -6,  0,  6,
          -4,  2,  8,
          -2,  4, 10
    }});

    m.Transpose(result);

    EXPECT_TRUE(expectedResult == result);
}

TEST(MatrixTest, matrix_transpose_non_square_i8)
{
    ml_tools::Matrix<int8_t, 4, 3> m({
        {  5,  6,  8,
           8, -7, -1,
          -2, -5,  8,
          -3,  1, -7,
    }});
    ml_tools::Matrix<int8_t, 3, 4> result;

    ml_tools::Matrix<int8_t, 3, 4> expectedResult({
        {  5,  8, -2, -3,
           6, -7, -5,  1,
           8, -1,  8, -7
    }});

    m.Transpose(result);

    EXPECT_TRUE(expectedResult == result);
}

TEST(MatrixTest, matrix_multiplication)
{

}

TEST(MatrixTest, matrix_inverse)
{

}

TEST(MatrixTest, matrix_dot_product)
{

}
