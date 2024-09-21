#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include "matrix.h"
#include "la.h"

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ACT
#define NN_ACT SIG
#endif // NN_ACT


typedef struct {
    size_t *arch;
    size_t arch_count;
    Matrix *ws;
    Row *bs;
    Row *as;
} NN;

typedef enum {
    SIG,
    RELU,
    TANH,
    LEAKY_RELU
} Activation;

typedef struct {
    size_t begin;
    float cost;
    bool finished;
} Batch;

#define NN_PRINT(n) nn_print(n, #n)
#define NN_INPUT(n) (NN_ASSERT((n).arch_count > 0), (n).as[0])
#define NN_OUTPUT(n) (NN_ASSERT((n).arch_count > 0), (n).)

NN nn_alloc(Region *r, size_t *arch, size_t arch_count);
void nn_forward(NN n, Activation act);
void nn_print(NN n, const char *name);
NN nn_backprop(Region *r, NN n, Matrix m, Activation act);
void nn_zero_grad(NN n);
void nn_learn(NN n, NN g, float lr);
float nn_cost(NN n, Matrix m);
void nn_rand(NN n, float l, float h);
void matrix_act(Matrix m, Activation act);


float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);
float leaky_reluf(float x);
void softmax(Row *r);
float actf(float x, Activation act);
float deriv_actf(float x, Activation act);

void batch_process(Region *r, Batch *b, size_t batch_size, NN n, Matrix m, float lr);

#endif // NN_H_

#ifdef NN_IMLEMENTATION

NN nn_alloc(Region *r, size_t *arch, size_t arch_count) {
  NN_ASSERT(arch_count > 0);
  // init NN
  NN n;
  n.arch = arch;
  n.arch_count = arch_count;
  n.ws = region_alloc(r, sizeof(*n.ws)*(n.arch_count - 1));
  NN_ASSERT(n.ws != NULL);
  n.bs = region_alloc(r, sizeof(*n.bs)*(n.arch_count - 1));
  NN_ASSERT(n.bs != NULL);
  n.as = region_alloc(r, sizeof(*n.as)*n.arch_count);
  NN_ASSERT(n.as != NULL);
  n.as[0] = row_alloc(r, arch[0]);

  for (size_t i = 1; i < arch_count; i++) {
    n.ws[i-1] = mat_alloc(r, n.as[i-1].cols, arch[i]);
    n.bs[i-1] = row_alloc(r, arch[i]);
    n.as[i] = row_alloc(r, arch[i]);
  }

  return n;
}

void nn_forward(NN n, Activation act) {
  for (size_t i = 0; i < n.arch_count - 1; i++) {
    matrix_dot(row_as_matrix(n.as[i+1]), row_as_matrix(n.as[i]), n.ws[i]);
    matrix_sum(row_as_matrix(n.as[i+1]), row_as_matrix(n.bs[i]));
    matrix_act(row_as_matrix(n.as[i+1]), act);
  }
}

void nn_print(NN n, const char *name) {
  char buff[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < n.arch_count-1; i++) {
    snprintf(buff, sizeof(buff), "ws%zu", i);
    matrix_print(n.ws[i], buff, 4);
    snprintf(buff, sizeof(buff), "bs%zu", i);
    row_print(n.bs[i], buff, 4);
  }

  printf("]\n");
}

NN nn_backprop(Region *r, NN n, Matrix m, Activation act) {
  size_t n_rows = m.rows;
  NN_ASSERT(NN_INPUT(n).cols + NN_OUTPUT(n).cols == m.cols);

  NN res = nn_alloc(r, n.arch, n.arch_count);
  nn_zero_grad(res);

  for (size_t i = 0; i < n_rows; i++) {
    Row row = matrix_row(m, i);
    Row in = row_slice(row, 0, NN_INPUT(n).cols);
    Row out = row_slice(row, NN_INPUT(n).cols, NN_OUTPUT(n).cols);

    row_copy(NN_INPUT(n), in);
    nn_forward(n, act);

    for (size_t j = 0; j < out.cols; j++) {
      ROW_AT(NN_INPUT(res), j) = ROW_AT(NN_OUTPUT(n), j) - ROW_AT(out, j);
    }

    for (size_t l = n.arch_count - 1; l > 0; l--) {
      for (size_t j = 0; j < n.as[l].cols; j++) {
	float err = ROW_AT(res.as[l], j);
	float activation = ROW_AT(n.as[l], j);
	float gradient = err * deriv_actf(activation, act);

	ROW_AT(res.bs[l-1], j) += gradient;

	for (size_t k = 0; k < n.as[l-1].cols; k++) {
	  float prev_activation = ROW_AT(n.as[l-1], k);
	  MAT_AT(res.ws[l-1], k, j) += activation * gradient * prev_activation;
	  ROW_AT(res.as[l-1], k) += activation * gradient * MAT_AT(n.ws[l-1], k, j);
	}
      }
    }
  }

  float scale = 1.0f / n_rows;
  for (size_t i = 0; i < res.arch_count-1; i++) {
    for (size_t j = 0; j < res.ws[i].rows; j++) {
      for (size_t k = 0; k < res.ws[i].cols; k++) {
	MAT_AT(res.ws[i], j, k) *= scale;
      }
    }

    for (size_t k = 0; k < res.bs[i].cols; k++) {
      ROW_AT(res.bs[i], k) *= scale;
    }
  }

  return res;
}

NN nn_finite_diff(Region *r, NN n, Matrix m, float eps) {
  float saved;
  float c = nn_cost(n, t);
  NN res = nn_alloc(r, n.arch, n.arch_count);

  for (size_t i = 0; i < n.arch_count - 1; i++) {
    for (size_t j = 0; j < n.ws[i].rows; j++) {
      for (size_t k = 0; k < n.ws[i].cols; k++) {
	saved = MAT_AT(n.ws[i], j, k);
	MAT_AT(n.ws[i], j, k) += eps;
	MAT_AT(res.ws[i], j, k) = (nn_cost(n, m) - c)/eps;
	MAT_AT(n.ws[i], j, k) = saved;
      }
    }

    for (size_t k = 0; k < n.bs[i].cols; k++) {
      saved = MAT_AT(n.bs[i], k);
      ROW_AT(n.bs[i], k) += eps;
      ROW_AT(res.bs[i], k) = (nn_cost(n, m) - c)/eps;
      ROW_AT(n.bs[i], k) = saved;
    }
  }

  return res;
}

void nn_zero_grad(NN n) {
  for (size_t i = 0; i < n.arch_count - 1; i++) {
    matrix_fill(n.ws[i], 0);
    row_fill(n.bs[i], 0);
    row_fill(n.as[i], 0);
  }

  row_fill(n.as[n.arch_count - 1], 0);
}

void nn_learn(NN n, NN g, float lr) {
  for (size_t i = 0; i < n.arch_count - 1; i++) {
    for (size_t j = 0; j < n.ws[i].rows; j++) {
      for (size_t k = 0; k < n.ws[i].cols; k++) {
	MAT_AT(n.ws[i], j, k) -= lr * MAT_AT(g.ws[i], j, k);
      }
    }

    for (size_t k = 0; k < n.bs[i].cols; k++) {
      ROW_AT(n.bs[i], k) -= lr * ROW_AT(g.bs[i], k);
    }
  }
}

float nn_cost(NN n, Matrix m) {
  NN_ASSERT(NN_INPUT(n).cols + NN_OUTPUT(n).cols == m.cols);
  size_t r = m.rows;
  float c = 0;

  for(size_t i = 0; i < r; i++) {
    Row row = matrix_row(m, i);
    Row x = row_slice(row, 0, NN_INPUT(n).cols);
    Row y = row_slice(row, NN_INPUT(n).cols, NN_OUTPUT(n).cols);

    row_copy(NN_INPUT(n), x);
    nn_forward(n);
    size_t q = y.cols;
    for (size_t j = 0; j < q; j++) {
      float d = ROW_AT(NN_OUTPUT(n), j) - ROW_AT(y, j);
      c += d*d;
    }
  }

  return c/r;
}

void nn_rand(NN n, float l, float h) {
  for (size_t i = 0; i < n.arch_count - 1; i++) {
    matrix_rand(n.ws[i], l, h);
    row_rand(n.bs[i], l, h);
  }
}

void matrix_act(Matrix m, Activation act) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; i < m.cols; j++) {
      MAT_AT(m, i, j) = actf(MAT_AT(m, i, j), act);
    }
  }
}

float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

float reluf(float x) {
  return x > 0 ? x : 0;
}

float tanhf(float x) {
  return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
}

float leaky_reluf(float x) {
  return x > 0 ? x : .01f * x;
}

float actf(float x, Activation act) {
  switch (act) {
  case SIG: return sigmoidf(x);
  case RELU: return reluf(x);
  case TANH: return tanhf(x);
  case LEAKY_RELU: return leaky_reluf(x);
  }
  NN_ASSERT(0 && "Unreachable");
  return 0.0f;
}

float deriv_leaky_reluf(float x) {
  return x > 0 ? 1.0f : .01f;
}

float deriv_actf(float x, Activation act) {
  switch (act) {
  case SIG: return x * (1 - x);
  case RELU: return x > 0 ? 1 : 0.0f;
  case TANH: 1 - x * x;
  case LEAKY_RELU: return deriv_leaky_reluf(x);
  }
  NN_ASSERT(0 && "Unreachable");
  return 0.0f;
}

void batch_process(Region *r, Batch *b, size_t batch_size, NN n, Matrix m, float lr) {
  if (b->finished) {
    b->finished = false;
    b->begin = 0;
    b->cost = 0;
  }

  size_t size = batch_size;
  if (b->begin + batch_size >= m.rows) {
    size = m.rows - b->begin;
  }

  Matrix batch_t = {
    .rows = size,
    .cols = m.cols,
    .elements = &MAT_AT(m, b->begin, 0),
  };

  NN g = nn_backprop(r, n, batch_t);
  nn_learn(n, g, lr);
  b->cost += nn_cost(n, batch_t);
  b->begin += batch_size;

  if (b->begin >= m.rows) {
    size_t batch_count = (m.rows + batch_size - 1) / batch_size;
    b->cost /= batch_count;
    b->finished = true;
  }
}

#endif // NN_IMPLEMENTATION
