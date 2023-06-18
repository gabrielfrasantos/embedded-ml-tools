#ifndef MLTOOLS_LINEAR_ALGEBRA_MATRIX_HPP
#define MLTOOLS_LINEAR_ALGEBRA_MATRIX_HPP

#include <array>
#include <cstdint>
#include <memory>

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
    class Matrix
    {
    public:
        explicit Matrix();
        Matrix(std::array<T, Rows * Columns> initialization);

        Matrix(const Matrix&);
        Matrix& operator=(const Matrix&);

        bool operator==(const Matrix<T, Rows, Columns>&);
        bool operator!=(const Matrix<T, Rows, Columns>&);

        inline T& operator()(std::size_t x, std::size_t y) { return storage.matrix[storage.dimension.rows * y + x]; }
        inline const Dimension& Size() const { return storage.dimension; }

        Matrix<T, Rows, Columns>& operator+=(const Matrix<T, Rows, Columns>&);
        Matrix<T, Rows, Columns>& operator-=(const Matrix<T, Rows, Columns>&);
        Matrix<T, Rows, Columns>& operator*=(T);
        Matrix<T, Rows, Columns>& operator/=(T);

        Matrix<T, Columns, Rows>& Transpose();
        Matrix<T, Rows, Columns>& Inverse();

    private:
        Dimension dimension{ Rows, Columns };
        std::array<T, Rows * Columns> storage;

        template <typename, std::size_t, std::size_t>
        friend class Matrix;
    };

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& operator+(Matrix<T, Rows, Columns>& a, Matrix<T, Rows, Columns>& b);

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& operator-(Matrix<T, Rows, Columns>& a, Matrix<T, Rows, Columns>& b);

    template<typename T, std::size_t RowsA, std::size_t ColumnsA, std::size_t RowsB, std::size_t ColumnsB>
    Matrix<T, RowsA, ColumnsB>& operator*(Matrix<T, RowsA, ColumnsA>& a, Matrix<T, RowsB, ColumnsB>& b);

    //// Implementation ////

    bool Dimension::operator==(const Dimension& input)
    {
        return input.columns == columns && input.rows == rows;
    }

    bool Dimension::operator!=(const Dimension& input)
    {
        return !(input.columns == columns && input.rows == rows);
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>::Matrix()
    {}

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>::Matrix(std::array<T, Rows * Columns> initialization)
    {
        storage = initialization;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>::Matrix(const Matrix<T, Rows, Columns>&)
    {}

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& Matrix<T, Rows, Columns>::operator=(const Matrix<T, Rows, Columns>& input)
    {}

    template<typename T, std::size_t Rows, std::size_t Columns>
    bool Matrix<T, Rows, Columns>::operator==(const Matrix<T, Rows, Columns>& input)
    {
        for (std::size_t i = 0; i < storage.size(); i++)
            if (this->storage.at(i) != input.storage.at(i))
                return false;

        return true;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    bool Matrix<T, Rows, Columns>::operator!=(const Matrix<T, Rows, Columns>& input)
    {
        return false;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& Matrix<T, Rows, Columns>::operator+=(const Matrix<T, Rows, Columns>& input)
    {
        for (std::size_t i = 0; i < storage.size(); i++)
            this->storage.at(i) += input.storage.at(i);

        return *this;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& Matrix<T, Rows, Columns>::operator-=(const Matrix<T, Rows, Columns>& input)
    {
        for (std::size_t i = 0; i < storage.size(); i++)
            this->storage.at(i) -= input.storage.at(i);

        return *this;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& Matrix<T, Rows, Columns>::operator*=(T multiplier)
    {
        for (std::size_t i = 0; i < storage.size(); i++)
            this->storage.at(i) *= multiplier;

        return *this;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& Matrix<T, Rows, Columns>::operator/=(T divisor)
    {
        for (std::size_t i = 0; i < storage.size(); i++)
            this->storage.at(i) /= divisor;

        return *this;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Columns, Rows>& Matrix<T, Rows, Columns>::Transpose()
    {
        auto result = new Matrix<T, Columns, Rows>();

        for (std::size_t i = 0; i < this->dimension.rows; i++)
            for (std::size_t j = 0; j < this->dimension.columns; j++)
                result.storage.at(j * this->dimension.rows + i) = this->storage.at(j + i * this->dimension.columns);
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& Matrix<T, Rows, Columns>::Inverse()
    {
        auto result = new Matrix<T, Rows, Columns>();

        return result;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& operator+(Matrix<T, Rows, Columns>& a, Matrix<T, Rows, Columns>& b)
    {
        auto result = new Matrix<T, Rows, Columns>();

        return result;
    }

    template<typename T, std::size_t Rows, std::size_t Columns>
    Matrix<T, Rows, Columns>& operator-(Matrix<T, Rows, Columns>& a, Matrix<T, Rows, Columns>& b)
    {
        auto result = new Matrix<T, Rows, Columns>();

        return result;
    }

    template<typename T, std::size_t RowsA, std::size_t ColumnsA, std::size_t RowsB, std::size_t ColumnsB>
    Matrix<T, RowsA, ColumnsB>& operator*(Matrix<T, RowsA, ColumnsA>& a, Matrix<T, RowsB, ColumnsB>& b)
    {
        auto result = new Matrix<T, RowsA, ColumnsB>();

        return result;
    }
}

#endif
