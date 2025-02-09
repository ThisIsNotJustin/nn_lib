#pragma once

#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdbool.h>
#include "../matrix/matrix.h"
#include "../la/la.h"
#include "../region/region.h"

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ACT
#define NN_ACT LEAKY_RELU
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
    LEAKY_RELU,
    SOFTMAX
} Activation;

typedef enum {
  MSE,
  BCE,
  CCE
} Loss;

typedef struct {
  Activation act;
  Loss loss;
} NNConfig;

typedef struct {
    size_t begin;
    float cost;
    bool finished;
} Batch;

#define NN_PRINT(n) nn_print(n, #n)
#define NN_INPUT(n) (NN_ASSERT((n).arch_count > 0), (n).as[0])
#define NN_OUTPUT(n) (NN_ASSERT((n).arch_count > 0), (n).as[(n).arch_count - 1])

NN nn_alloc(Region *r, size_t *arch, size_t arch_count);
void nn_forward(NN n, Activation act);
void nn_print(NN n, const char *name);
NN nn_backprop(Region *r, NN n, Matrix m, NNConfig config);
NN nn_finite_diff(Region *r, NN n, Matrix m, float eps);
void nn_zero_grad(NN n);
void nn_learn(NN n, NN g, float lr);
float nn_cost(NN n, Matrix m, Loss loss);
void nn_rand(NN n, float l, float h);
void matrix_act(Matrix m, Activation act);


float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);
float leaky_reluf(float x);
void softmax(Matrix m);
float actf(float x, Activation act);
float deriv_actf(float x, Activation act);
float deriv_loss(float y_pred, float y_true, Loss loss);

void batch_process(Region *r, Batch *b, size_t batch_size, NN n, Matrix m, float lr, NNConfig config);

float compute_mse(NN n, Matrix m);
float compute_bce(NN n, Matrix m);
float compute_cce(NN n, Matrix m);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

NN nn_alloc(Region *r, size_t *arch, size_t arch_count) {
  NN_ASSERT(arch_count > 0);
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

  for (size_t i = 1; i < arch_count; ++i) {
    n.ws[i-1] = matrix_alloc(r, n.as[i-1].cols, arch[i]);
    n.bs[i-1] = row_alloc(r, arch[i]);
    n.as[i] = row_alloc(r, arch[i]);
  }

  return n;
}

void nn_forward(NN n, Activation act) {
  for (size_t i = 0; i < n.arch_count - 1; ++i) {
    matrix_dot(row_as_matrix(n.as[i+1]), row_as_matrix(n.as[i]), n.ws[i]);
    matrix_add(row_as_matrix(n.as[i+1]), row_as_matrix(n.bs[i]));

    if (i == n.arch_count - 2 && act == SOFTMAX) {
      softmax(row_as_matrix(n.as[i + 1]));
    } else {
      matrix_act(row_as_matrix(n.as[i+1]), act);
    }
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

NN nn_backprop(Region *r, NN n, Matrix m, NNConfig config) {
  size_t n_rows = m.rows;
  NN_ASSERT(NN_INPUT(n).cols + NN_OUTPUT(n).cols == m.cols);

  NN res = nn_alloc(r, n.arch, n.arch_count);
  nn_zero_grad(res);

  for (size_t i = 0; i < n_rows; ++i) {
    Row row = matrix_row(m, i);
    Row in = row_slice(row, 0, NN_INPUT(n).cols);
    Row out = row_slice(row, NN_INPUT(n).cols, NN_OUTPUT(n).cols);

    row_copy(NN_INPUT(n), in);
    nn_forward(n, config.act);

    for (size_t j = 0; j < n.arch_count; ++j) {
      row_fill(res.as[j], 0);
    }
    for (size_t j = 0; j < out.cols; ++j) {
      ROW_AT(res.as[n.arch_count - 1], j) = (ROW_AT(NN_OUTPUT(n), j) - ROW_AT(out, j));
    }

    for (size_t l = n.arch_count - 1; l > 0; --l) {
      for (size_t j = 0; j < n.as[l].cols; ++j) {
        float activation = ROW_AT(n.as[l], j);
        float err = ROW_AT(res.as[l], j);
        float dact = deriv_actf(activation, config.act);

        if (l == n.arch_count - 1) {
          err = deriv_loss(activation, ROW_AT(out, j), config.loss);
        }
        
        ROW_AT(res.bs[l - 1], j) += dact * err;
        for (size_t k = 0; k < n.as[l - 1].cols; ++k) {
          float prev_activation = ROW_AT(n.as[l - 1], k);
          float w = MAT_AT(n.ws[l - 1], k, j);
          MAT_AT(res.ws[l - 1], k, j) += err * dact * prev_activation;
          ROW_AT(res.as[l - 1], k) += err * dact * w;
        }
      }
    }
  }

  for (size_t i = 0; i < res.arch_count-1; ++i) {
    for (size_t j = 0; j < res.ws[i].rows; ++j) {
      for (size_t k = 0; k < res.ws[i].cols; ++k) {
	      MAT_AT(res.ws[i], j, k) /= n_rows;
      }
    }

    for (size_t k = 0; k < res.bs[i].cols; ++k) {
      ROW_AT(res.bs[i], k) /= n_rows;
    }
  }

  printf("Backprop complete\n");
  return res;
}

NN nn_finite_diff(Region *r, NN n, Matrix m, float eps) {
  float saved;
  // need loss float c = nn_cost(n, m);
  // printf("Initial cost: %f\n", c);
  NN res = nn_alloc(r, n.arch, n.arch_count);

  for (size_t i = 0; i < n.arch_count - 1; ++i) {
    printf("Layer %zu\n", i);
    for (size_t j = 0; j < n.ws[i].rows; ++j) {
      for (size_t k = 0; k < n.ws[i].cols; ++k) {
        printf("still doing stuff\n");
        saved = MAT_AT(n.ws[i], j, k);
        MAT_AT(n.ws[i], j, k) += eps;
        // need loss MAT_AT(res.ws[i], j, k) = (nn_cost(n, m) - c)/eps;
        MAT_AT(n.ws[i], j, k) = saved;
      }
    }

    for (size_t k = 0; k < n.bs[i].cols; ++k) {
      saved = ROW_AT(n.bs[i], k);
      ROW_AT(n.bs[i], k) += eps;
      // need loss ROW_AT(res.bs[i], k) = (nn_cost(n, m) - c)/eps;
      ROW_AT(n.bs[i], k) = saved;
    }
  }

  printf("Finite differences computed.\n");
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
  for (size_t i = 0; i < n.arch_count - 1; ++i) {
    for (size_t j = 0; j < n.ws[i].rows; ++j) {
      for (size_t k = 0; k < n.ws[i].cols; ++k) {
	      MAT_AT(n.ws[i], j, k) -= lr * MAT_AT(g.ws[i], j, k);
      }
    }

    for (size_t k = 0; k < n.bs[i].cols; k++) {
      ROW_AT(n.bs[i], k) -= lr * ROW_AT(g.bs[i], k);
    }
  }
}

/*

float nn_cost(NN n, Matrix m, Loss loss) {
  NN_ASSERT(NN_INPUT(n).cols + NN_OUTPUT(n).cols == m.cols);
  size_t r = m.rows;
  float cost = 0;

  for (size_t i = 0; i < r; ++i) {
    Row row = matrix_row(m, i);
    Row x = row_slice(row, 0, NN_INPUT(n).cols);
    Row y = row_slice(row, NN_INPUT(n).cols, NN_OUTPUT(n).cols);

    row_copy(NN_INPUT(n), x);
    nn_forward(n, NN_ACT);

    for (size_t j = 0; j < y.cols; ++j) {
      float y_pred = ROW_AT(NN_OUTPUT(n), j);
      y_pred = y_pred < 1e-7f ? 1e-7f : (y_pred > 1-1e-7f ? 1-1e-7f : y_pred);
      float y_true = ROW_AT(y, j);
      cost -= y_true * logf(y_pred) + (1 - y_true) * logf(1 - y_pred);
    }
  }

  return cost / r;
}

*/

float nn_cost(NN n, Matrix m, Loss loss) {
  switch (loss) {
    case MSE:
      return compute_mse(n, m);
    case BCE:
      return compute_bce(n, m);
    case CCE:
      return compute_cce(n, m);
    default:
      printf("Error: Unknown loss function\n");
      exit(1);
  }
}

float compute_mse(NN n, Matrix m) {
  float sum = 0.0f;
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      float y_true = MAT_AT(m, i, j);
      float y_pred = MAT_AT(n.as[n.arch_count - 1], i, j);
      float diff = y_true - y_pred;

      sum += diff * diff;
    }
  }

  return sum / (m.rows * m.cols);
}

float compute_bce(NN n, Matrix m) {
  const float epsilon = 1e-9f;
  float sum = 0.0f;
  
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      float y_true = MAT_AT(m, i, j);
      float y_pred = MAT_AT(n.as[n.arch_count - 1], i, j);
      y_pred = fmaxf(epsilon, fminf(1.0f - epsilon, y_pred));

      sum += y_true * logf(y_pred) + (1 - y_true) * logf(1 - y_pred);
    }
  }
  
  return -sum / (m.rows * m.cols);
}

float compute_cce(NN n, Matrix m) {
  const float epsilon = 1e-9f;
  float sum = 0.0f;

  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      float y_true = MAT_AT(m, i, j);
      if (y_true > 0.5f) {
        float y_pred = MAT_AT(n.as[n.arch_count - 1], i, j);
        y_pred = fmaxf(epsilon, y_pred);
        sum += logf(y_pred);
      }
    }
  }
  
  return -sum / m.rows;
}

void nn_rand(NN n, float l, float h) {
  for (size_t i = 0; i < n.arch_count - 1; ++i) {
    matrix_rand(n.ws[i], l, h);
    row_rand(n.bs[i], l, h);
  }
}

void matrix_act(Matrix m, Activation act) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
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
    case SOFTMAX: return x;
  }
  NN_ASSERT(0 && "Unreachable");
  return 0.0f;
}

void softmax(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
        float max = MAT_AT(m, i, 0);
        for (size_t j = 1; j < m.cols; ++j) {
            if (MAT_AT(m, i, j) > max) {
                max = MAT_AT(m, i, j);
            }
        }

        float sum = 0.0;
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = exp(MAT_AT(m, i, j) - max);
            sum += MAT_AT(m, i, j);
        }

        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) /= sum;
        }
    }
}

float deriv_leaky_reluf(float x) {
  return x > 0 ? 1.0f : .01f;
}

float deriv_actf(float x, Activation act) {
  switch (act) {
    case SIG: return x * (1 - x);
    case RELU: return x > 0 ? 1 : 0.0f;
    case TANH: return 1 - x * x;
    case LEAKY_RELU: return deriv_leaky_reluf(x);
    case SOFTMAX: return 1.0f;
  }
  NN_ASSERT(0 && "Unreachable");
  return 0.0f;
}

void batch_process(Region* r, Batch* b, size_t batch_size, NN n, Matrix m, float lr, NNConfig config) {
  printf("Entering batch_process\n");
  printf("Initial batch state: finished=%d, begin=%zu, cost=%f\n", b->finished, b->begin, b->cost);
  if (b->finished) {
    b->finished = false;
    b->begin = 0;
    b->cost = 0;
  }

  if (b->begin >= m.rows) {
    printf("Error: batch begin %zu is out of bounds for matrix rows %zu", b->begin, m.rows);
    exit(1);
  }

  size_t size = batch_size;
  if (b->begin + batch_size >= m.rows) {
    size = m.rows - b->begin;
    printf("Adjusted batch size to %zu\n", size);
  }

  printf("Creating batch_t matrix: rows=%zu, cols=%zu, begin=%zu\n", size, m.cols, b->begin);
  if (size == 0 || m.cols == 0) {
    printf("Error: Invalid batch_t dimensions: rows=%zu, cols=%zu\n", size, m.cols);
  }

  Matrix batch_t = {
    .rows = size,
    .cols = m.cols,
    .elements = &MAT_AT(m, b->begin, 0),
  };

  if (batch_t.elements == NULL) {
    printf("Error: batch_t elements pointer is NULL\n");
  }

  printf("Starting backprop\n");
  NN g = nn_backprop(r, n, batch_t, config);
  // NN g = nn_finite_diff(r, n, batch_t, 1e-5);
  if (g.arch_count != n.arch_count) {
    printf("Error: Gradient NN structure, g doesn't match original NN, n\n");
  }

  nn_learn(n, g, lr);
  b->cost += nn_cost(n, batch_t, config.loss);
  printf("Batch cost: %f\n", b->cost);
  b->begin += batch_size;
  printf("Updated batch begin: %zu\n", b->begin);

  if (b->begin >= m.rows) {
    size_t batch_count = (m.rows + batch_size - 1) / batch_size;
    b->cost /= batch_count;
    b->finished = true;
    printf("Batch processing finished. Final average cost: %f\n", b->cost);
  }
}

float deriv_loss(float y_pred, float y_true, Loss loss) {
  switch (loss) {
    case MSE:
      return 2 * (y_pred - y_true);

    case BCE:
      y_pred = y_pred < 1e-7f ? 1e-7f : (y_pred > 1-1e-7f ? 1-1e-7f : y_pred);
      return (y_pred - y_true) / (y_pred * (1 - y_pred));

    case CCE:
      return y_pred - y_true;

    default:
      NN_ASSERT(0 && "Unreachable");
      return 0.0f;
  }
}

#endif // NN_IMPLEMENTATION
