#ifndef MLTOOLS_LINEAR_ALGEBRA_MATRIX_HPP
#define MLTOOLS_LINEAR_ALGEBRA_MATRIX_HPP

#include <array>
#include <cstdint>
#include "infra/util/BoundedVector.hpp"
#include "infra/util/WithStorage.hpp"

namespace ml_tools
{
    struct Dimension
    {
        std::size_t rows;
        std::size_t columns;

        bool operator==(const Dimension&);
        bool operator!=(const Dimension&);
    };

    template<typename T, std::size_t Rows, std::size_t Columns>
    struct MatrixStorage
    {
        Dimension dimension{ Rows, Columns };
        std::array<T, Rows * Columns> matrix;
    };

    template<typename T>
    class Matrix
    {
    public:
        template<std::size_t Rows, std::size_t Columns>
        using WithRowsAndColumns = infra::WithStorage<Matrix<T>, MatrixStorage<T, Rows, Columns>>;

        explicit Matrix();
        Matrix(const infra::BoundedVector<T>& initialization);

        Matrix(const Matrix&);
        Matrix& operator=(const Matrix&);

        bool operator==(const Matrix<T>&);
        bool operator!=(const Matrix<T>&);

        inline T& operator()(std::size_t x, std::size_t y) { return storage.matrix[storage.dimension.rows * y + x]; }
        inline const Dimension& Size() const { return storage.dimension; }

        Matrix<T>& operator+=(const Matrix<T>&);
        Matrix<T>& operator-=(const Matrix<T>&);
        Matrix<T>& operator*=(T);
        Matrix<T>& operator/=(T);

        void Multiply(const Matrix<T>& input, Matrix<T>& result);
        void Transpose(Matrix<T>& result);
        void Inverse(Matrix<T>& result);

        static void CreateIdentity(Matrix<T>& result);
        static T DotProduct(Matrix<T>& a, Matrix<T>& b);

    private:
        MatrixStorage& storage;
    };

    //// Implementation ////
    
    bool Dimension::operator==(const Dimension& input)
    {
        return input.columns == columns && input.rows == rows;
    }

    bool Dimension::operator!=(const Dimension& input)
    {
        return !(input.columns == columns && input.rows == rows);
    }

    template<typename T>
    Matrix<T>::Matrix()
    {}

    template<typename T>
    Matrix<T>::Matrix(const infra::BoundedVector<T>& initialization)
    {}

    template<typename T>
    Matrix<T>::Matrix(const Matrix<T>&)
    {}

    template<typename T>
    Matrix<T>& Matrix<T>::operator=(const Matrix<T>& input)
    {}

    template<typename T>
    bool Matrix<T>::operator==(const Matrix<T>& input)
    {
        return false;
    }

    template<typename T>
    bool Matrix<T>::operator!=(const Matrix<T>& input)
    {
        return false;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator+=(const Matrix<T>&)
    {
        return this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator-=(const Matrix<T>&)
    {
        return this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator*=(T)
    {
        return this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator/=(T)
    {
        return this;
    }

    template<typename T>
    void Matrix<T>::Multiply(const Matrix<T>& input, Matrix<T>& result)
    {

    }

    template<typename T>
    void Matrix<T>::Transpose(Matrix<T>& result)
    {
        
    }

    template<typename T>
    void Matrix<T>::Inverse(Matrix<T>& result)
    {
        
    }
}

#endif
