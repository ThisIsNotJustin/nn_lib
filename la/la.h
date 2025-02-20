#pragma once

#ifndef LA_H_
#define LA_H_

#include "../matrix/matrix.h"
#include "../nn/nn.h"
#include <math.h>

float matrix_mean(Matrix m);
void matrix_reshape(Matrix dst, Matrix a, size_t new_rows, size_t new_cols);
void matrix_transpose(Matrix dst, Matrix a);
void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_add(Matrix a, Matrix b);
void matrix_subtract(Matrix a, Matrix b);
void matrix_scale(Matrix m, float n);
void matrix_add_scalar(Matrix m, float n);
float matrix_fnorm(Matrix m);
Matrix scaled_dot_product(Region *r, Matrix *Q, Matrix *K, Matrix *V);

void softmax(Matrix m);

// TODO:
// void matrix_flatten(Matrix dst, Matrix a);
// Matrix matrix_inverse(const Matrix *m);
// float matrix_det(const Matrix *m);
// Matrix matrix_solve(const Matrix *m);
// float* matrix_eigenvalues(const Matrix *m, size_t *ev);
// Matrix* matrix_eigenvectors(const Matrix *m, Matrix *ev);

#endif // LA_H_

#ifdef LA_IMPLEMENTATION

/*
  Calculates the mean of matrix elements
    
  Parameters:
    m - Input matrix
    
  Returns:
    Mean of all elements as float
*/
float matrix_mean(Matrix m) {
  float sum = 0.0f;
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      sum += MAT_AT(m, i, j);
    }
  }

  return sum / (m.rows * m.cols);
}

/*
  Computes matrix dot product
    
  Parameters:
    dst - Destination matrix
    a - Left operand matrix
    b - Right operand matrix
    
  Preconditions:
    a.cols == b.rows
    dst.rows == a.rows && dst.cols == b.cols
*/
void matrix_dot(Matrix dst, Matrix a, Matrix b) {
  MAT_ASSERT(a.cols == b.rows);
  size_t c = a.cols;
  MAT_ASSERT(dst.rows == a.rows);
  MAT_ASSERT(dst.cols == b.cols);

  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < c; k++) {
	      MAT_AT(dst, i , j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
  
}

/*
  Scales matrix elements by a factor, n
    
  Parameters:
    m - Matrix being scaled
    n - Scalar multiplier
*/
void matrix_scale(Matrix m, float n) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) *= n;
    }
  }
}

/*
  Adds a scalar value to all matrix elements
    
  Parameters:
    m - Matrix being modified
    n - Scalar value
*/
void matrix_add_scalar(Matrix m, float n) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) += n;
    }
  }
}

/*
  Performs in-place matrix addition
    
  Parameters:
    a - Matrix being modified
    b - Matrix to add by
    
  Preconditions:
    a.rows == b.rows && a.cols == b.cols
*/
void matrix_add(Matrix a, Matrix b) {
  MAT_ASSERT(a.rows == b.rows);
  MAT_ASSERT(a.cols == b.cols);

  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(a, i, j) += MAT_AT(b, i, j);
    }
  }
}

/*
  Performs in-place matrix subtraction
    
  Parameters:
    a - Matrix to modify
    b - Matrix to subtract by
    
  Preconditions:
    a.rows == b.rows && a.cols == b.cols
*/
void matrix_subtract(Matrix a, Matrix b) {
  MAT_ASSERT(a.rows == b.rows);
  MAT_ASSERT(a.cols == b.cols);

  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(a, i, j) -= MAT_AT(b, i, j);
    }
  }
}

/*
    Reshapes matrix to new dimensions
    
    Parameters:
        dst - Destination matrix (must be preallocated)
        a - Source matrix to reshape
        new_rows - Target row count for reshaped matrix
        new_cols - Target column count for reshaped matrix
    
    Preconditions:
      dst.rows == new_rows && dst.cols == new_cols
*/
void matrix_reshape(Matrix dst, Matrix a, size_t new_rows, size_t new_cols) {
  MAT_ASSERT(dst.rows == new_rows);
  MAT_ASSERT(dst.cols == new_cols);

  for (size_t i = 0; i < new_rows * new_cols; i++) {
    dst.elements[i] = a.elements[i];
  }
}

/*
  Transposes matrix dimensions
    
  Parameters:
    dst - Destination matrix (must be preallocated)
    a - Source matrix to transpose
    
  Preconditions:
    dst.rows == a.cols && dst.cols == a.rows
*/
void matrix_transpose(Matrix dst, Matrix a) {
  MAT_ASSERT(dst.rows == a.cols);
  MAT_ASSERT(dst.cols == a.rows);

  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(dst, j, i) = MAT_AT(a, i, j);
    }
  }
}

/*
  Computes fnorm of matrix
    
  Parameters:
    m - Input matrix
    
  Returns:
    Square root of sum of squared elements
*/
float matrix_fnorm(Matrix m) {
  float sum = 0.0f;
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      sum += MAT_AT(m, i, j) * MAT_AT(m, i, j);
    }
  }

  return sqrt(sum);
}

/*
  Transformer Scaled Dot-Product Attention

  attention(Q, K, V) = softmax(QKT/sqrt(dk)) V 
    
  Parameters:
    Requires a Region due to allocating more memory for temporary
    and resultant matrices
    Matrix Q - Queries, K - Keys, and V - Values

  Returns:
    new Matrix of softmax(QKT/sqrt(dk)) V 
*/
Matrix scaled_dot_product(Region *r, Matrix *Q, Matrix *K, Matrix *V) {
  MAT_ASSERT(Q->cols == K->cols);
  MAT_ASSERT(K->rows == V->rows);

  size_t m = Q->rows;
  size_t n = K->rows;
  size_t d_k = Q->cols;

  // should we overwrite the memory region of K for K_T
  // and Q for scores
  // rather than allocating more memory within this function?
  Matrix K_T = matrix_alloc(r, K->cols, K->rows);
  matrix_transpose(K_T, *K);
  Matrix scores = matrix_alloc(r, m, n);
  matrix_dot(scores, *Q, K_T);

  float scale = 1.0f / sqrtf((float)d_k);
  matrix_scale(scores, scale);

  softmax(scores);

  // again should memory simply be overwritten rather
  // than allocating more??
  Matrix res = matrix_alloc(r, m, V->cols);
  matrix_dot(res, scores, *V);

  return res;
}

/*
  something about cache locality
  dividing the matrix in to smaller blocks (16x16)
  same big-o time complexity but reuses mlmm variable
  dst matrix is overwritten so keep that in mind

  Parameters:
    Matrix *dst - destination Matrix, Matrix *m - Matrix 1
    Matrix *n - Matrix 2

  Accepts a destination Matrix rather than returning a new Matrix

*/
void matrix_mul(Matrix *dst, Matrix *m, Matrix *n) {
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < n->cols; j++) {
      dst->elements[i * n->cols + j] = 0.0f;
    }
  }

  const size_t block_size = 16;
  for (size_t i = 0; i < m->rows; i += block_size) {
    for (size_t k = 0; k < m->cols; k += block_size) {
      for (size_t j = 0; j < n->cols; j += block_size) {
        for (size_t l = i; l < i + block_size && l < m->rows; l++) {
          for (size_t mm = k; mm < k + block_size && mm < m->cols; mm++) {
            const float mlmm = m->elements[l * m->cols + mm];
            for (size_t nn = j; nn < j + block_size && nn < n->cols; nn++) {
              dst->elements[l * n->cols + nn] += mlmm * n->elements[mm * n->cols + nn];
            }
          }
        }
      }
    }
  }
}

#endif // LA_IMPLEMENTATION
