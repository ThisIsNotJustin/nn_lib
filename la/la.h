#pragma once

#ifndef LA_H_
#define LA_H_

#include "../matrix/matrix.h"
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
Matrix* scaled_dot_product(Matrix *Q, Matrix *K, Matrix *V);

// TODO:
// void matrix_flatten(Matrix dst, Matrix a);
// Matrix matrix_inverse(const Matrix *m);
// float matrix_det(const Matrix *m);
// Matrix matrix_solve(const Matrix *m);
// float* matrix_eigenvalues(const Matrix *m, size_t *ev);
// Matrix* matrix_eigenvectors(const Matrix *m, Matrix *ev);

#endif // LA_H_

#ifdef LA_IMPLEMENTATION

float matrix_mean(Matrix m) {
  float sum = 0.0f;
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      sum += MAT_AT(m, i, j);
    }
  }

  return sum / (m.rows * m.cols);
}

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

void matrix_scale(Matrix m, float n) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) *= n;
    }
  }
}

void matrix_add_scalar(Matrix m, float n) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) += n;
    }
  }
}

void matrix_add(Matrix a, Matrix b) {
  MAT_ASSERT(a.rows == b.rows);
  MAT_ASSERT(a.cols == b.cols);

  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(a, i, j) += MAT_AT(b, i, j);
    }
  }
}

void matrix_subtract(Matrix a, Matrix b) {
  MAT_ASSERT(a.rows == b.rows);
  MAT_ASSERT(a.cols == b.cols);

  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(a, i, j) -= MAT_AT(b, i, j);
    }
  }
}

void matrix_reshape(Matrix dst, Matrix a, size_t new_rows, size_t new_cols) {
  MAT_ASSERT(dst.rows == new_rows);
  MAT_ASSERT(dst.cols == new_cols);

  for (size_t i = 0; i < new_rows * new_cols; i++) {
    dst.elements[i] = a.elements[i];
  }
}

void matrix_transpose(Matrix dst, Matrix a) {
  MAT_ASSERT(dst.rows == a.cols);
  MAT_ASSERT(dst.cols == a.rows);

  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      MAT_AT(dst, j, i) = MAT_AT(a, i, j);
    }
  }
}

float matrix_fnorm(Matrix m) {
  float sum = 0.0f;
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      sum += MAT_AT(m, i, j) * MAT_AT(m, i, j);
    }
  }

  return sqrt(sum);
}

Matrix* scaled_dot_product(Matrix *Q, Matrix *K, Matrix *V) {
  MAT_ASSERT(Q->cols == K->cols);
  MAT_ASSERT(K->rows == V->rows);

  size_t m = Q->rows;
  size_t n = K->rows;
  size_t d_k = Q->cols;

  Matrix K_T = matrix_alloc(NULL, K->cols, K->rows);
  matrix_transpose(K_T, *K);
  Matrix scores = matrix_alloc(NULL, m, n);
  matrix_dot(scores, *Q, K_T);

  float scale = 1.0f / sqrtf((float)d_k);
  matrix_scale(scores, scale);

  softmax(scores);

  Matrix *res = malloc(sizeof(Matrix));
  *res = matrix_alloc(NULL, m, V->cols);
  matrix_dot(*res, scores *V);

  free(K_T.elements);
  free(scores.elements);

  return res;
}

#endif // LA_IMPLEMENTATION
