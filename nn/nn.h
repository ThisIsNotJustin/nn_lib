#pragma once

/*
  TODO: 
    Ensure transformers work well with larger datasets
    I believe we are missing a backprop method for transformers?
    How did you mess that up??

    Refactor code for better performance with pointers and
    references when possible

    Refactor structs and functions to be more easily reused
    At the moment we have feed forward and transformer
    yet transformers need their own feed forward logic??
    We need common ground for the code

    I'd imagine implementing convolutional neural nets
    in this current style would quite literally double the codebase
    this needs fixed

  IDEAS:
    typedef enum {
      DENSE_LAYER,
      ATTENTION_LAYER,
      CONV_LAYER
    } LayerType;

    typedef struct {
      LayerType type;
      union {
        struct { Matrix W; Row b; } dense;
        struct { Matrix Wq, Wk, Wv, Wo; } attention;
        struct { FilterBank filters; } conv;
      };
      // shared
      Matrix output;
      Matrix gradients;
    } Layer;

    typedef struct {
      Layer *layers;
      size_t num_layers;
      Region *memory_region;
    } Network;

    // shared
    typedef struct {
      Matrix (*forward)(Layer *layer, Matrix input);
      Matrix (*backward)(Layer *layer, Matrix grad_output);
      void (*update)(Layer *layer, float lr);
    } LayerOps;

    Network create_transformer(Region *r, TConfig cfg) {
      Network n = initalize it
      add_attention_layer(&n, cfg);
      add_ffn_layer(&n, cfg);
      add_norm_layer(&n, cfg);
      return n;
    }

    void train_step(Network *n, Matrix batch, float lr) {
      Matrix output = forward_pass(n, batch);
      Matrix grad = loss_gradient(output, targets);
      backward_pass(n, grad);
      update_parameters(n, lr);
    }

    Need Optimizer (Adam, SGD)
*/

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

/*
  Basic Neural Network Structure
  Contains achitecture dimensions, weights, biases, and activations

  Fields:
    arch - array for the dimensions (ie 784, 64, 32, 10)
    arch_count - num layers in the neural network
    ws - Matrix of Weights
    bs - Row Vector of biases per layer
    as - Row Vector of activations per layer
*/
typedef struct {
    size_t *arch;
    size_t arch_count;
    Matrix *ws;
    Row *bs;
    Row *as;
} NN;

/*
  Activation Function Enum
  Supported Activations:
    SIG - Sigmoid
    RELU - Rectified Linear Unit
    TANH - Hyperbolic Tangent
    LEAKY_RELU - Prevent Dying ReLU
    SOFTMAX - Softmax for output layers
*/
typedef enum {
    SIG,
    RELU,
    TANH,
    LEAKY_RELU,
    SOFTMAX
} Activation;

/*
  Loss Function Enum
  Supported Loss Functions:
    MSE - Mean Squared Error
    BCE - Binary Cross Entropy
    CCE - Categorical Cross Entropy
*/
typedef enum {
  MSE,
  BCE,
  CCE
} Loss;

/*
  Neural Network Config
  Contains both Activation and Loss Enums

  Fields:
    act - Activation function for hidden layers
    loss - Loss function for training
*/
typedef struct {
  Activation act;
  Loss loss;
} NNConfig;

/*
  Transformer Config
  This and all other transformer aspects are a work 
  in progress. Currently unused

  Fields:
    act - Activation function for hidden layers
    loss - Loss function for training
*/
typedef struct {
  size_t layers;
  size_t *arch;
  size_t att_heads;
  size_t *ff_arch;
} TConfig;

/*
  Multi Attention Head


  Fields:
    act - Activation function for hidden layers
    loss - Loss function for training
*/
typedef struct {
  Matrix *Wq;
  Matrix *Wk;
  Matrix *Wv;
  Matrix *Wo;
  Matrix *Q;
  Matrix *K;
  Matrix *V;
  Matrix *scores;
  size_t att_heads;
  size_t *arch;
  size_t arch_count;
} AttentionHead;

typedef struct {
  Matrix *W1;
  Row *b1;
  Matrix *W2;
  Row *b2;
} FeedForward;

typedef struct {
  AttentionHead att;
  FeedForward ff;
  Matrix *norm1;
  Matrix *norm2;
  Matrix *norm1_input;
  Matrix *norm2_input;
  Matrix *ff_input;
  Matrix *ff_hidden;
} TransformerLayer;

typedef struct {
  TransformerLayer *tlayers;
  size_t layers;
  Matrix *encode;
  size_t *arch;
} Transformer;

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

Matrix attention_forward(Region *r, AttentionHead *mha, Matrix *m);
Matrix tlayer_forward(Region *r, TransformerLayer *tlayer, Matrix *m);
Matrix transformer_forward(Region *r, Transformer *t, Matrix *m);
Matrix feed_forward(Region *r, FeedForward *ff, Matrix *m);

Transformer* transformer_alloc(Region *r, TConfig config);
TransformerLayer* tlayer_alloc(Region *r, size_t d_model, size_t d_ff, size_t heads);
void transformer_backprop(Region *r, Transformer *t, Transformer *grad_t, Matrix *in, Matrix *grad);
Matrix* norm_backward(Region *r, Matrix *grad, Matrix *x_forward, Matrix *gamma);
Matrix* ff_backward(Region *r, FeedForward *ff, Matrix *grad, Matrix *input, Matrix *hidden);
Matrix* attention_backward(Region *r, AttentionHead *mha, Matrix *grad, Matrix *Q, Matrix *K, Matrix *V, Matrix *scores);
void transformer_learn();

Matrix split_heads(Region *r, Matrix *m, AttentionHead *mha);
Matrix concat_heads(Region *r, Matrix *m, AttentionHead *mha);
Matrix layer_norm(Region *r, Matrix *m, Matrix *norm_p);
void add_bias(Matrix m, Row b);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

/*
  Allocate neural network from memory region
    
  Parameters:
    r - Memory region for allocation
    arch - Array defining size of architecture
    arch_count - Number of layers
    
  Returns:
    Initialized NN structure with allocated matrices
*/
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

/*
  Perform forward pass through network
    
  Parameters:
    n - Neural Network structure
    act - Activation function to use
*/
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

/*
  Print network architecture and parameters
    
  Parameters:
    n - Neural Network to print
    name - Label to display
*/
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

/*
  Backpropagation to compute gradients
    
  Parameters:
    r - Memory region for temporary allocations
    n - Neural Network to train
    m - Training data matrix
    config - Network configuration for activations and loss
    
  Returns:
    NN structure containing computed gradients
*/
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

/*
  Finite differences function
  No reason to use outside of learning and testing

  Just use backprop
  
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
*/

/*
  Set entire Neural Network to 0
    
  Parameters:
    n - Neural Network to set to 0
*/
void nn_zero_grad(NN n) {
  for (size_t i = 0; i < n.arch_count - 1; i++) {
    matrix_fill(n.ws[i], 0);
    row_fill(n.bs[i], 0);
    row_fill(n.as[i], 0);
  }

  row_fill(n.as[n.arch_count - 1], 0);
}

/*
  Update (learn) network parameters using gradients
    
  Parameters:
    n - Neural network to update
    g - Gradients computed from backprop
    lr - Learning rate
*/
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
  Older implementation of nn_cost
  likely need to delete but what if it could be used
  again?? delulu but rather not delete it

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

/*
  Compute cost over training data
    
  Parameters:
    n - Neural network
    m - Training data matrix
    loss - Loss function to use
    
  Returns:
    Computed loss value
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

/*
  Compute Mean Squared Error
    
  Parameters:
    n - Neural network
    m - Matrix of target values
    
  Returns:
    The Mean Squared Error
*/
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

/*
  Compute Binary Cross Entropy
    
  Parameters:
    n - Neural network
    m - Matrix of target values
    
  Returns:
    Binary Cross Entropy
*/
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

/*
  Compute Categorical Cross Entropy

  Parameters:
    n - Neural Network
    m - Matrix of target values

  Returns:
    Categorical Cross Entropy
*/
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

/*
  Randomize Neural Network Parameters

  Parameters:
    n - Neural Network
    l - lower bound for random values
    h - upper bound for random values
*/
void nn_rand(NN n, float l, float h) {
  for (size_t i = 0; i < n.arch_count - 1; ++i) {
    matrix_rand(n.ws[i], l, h);
    row_rand(n.bs[i], l, h);
  }
}

/*
  Apply activation function to given Matrix

  Parameters:
    m - Matrix
    act - Activation function
*/
void matrix_act(Matrix m, Activation act) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = actf(MAT_AT(m, i, j), act);
    }
  }
}

/*
  Sigmoid Activation Function
    
  Parameters:
    x - Input value
    
  Returns:
    Sigmoid(x) = 1 / (1 + exp(-x))
*/
float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

/*
  ReLU Activation Function

  Parameters:
    x - Input value

  Returns ReLU(x) = 0 or x
*/
float reluf(float x) {
  return x > 0 ? x : 0;
}

/*
  Tanh Activation Function

  Parameters:
    x - Input value

  Returns:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
*/
float tanhf(float x) {
  return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
}

/*
  Leaky ReLU activation function
    
  Parameters:
    x - Input value
    
  Returns:
    LeakyReLU(x) = x if x > 0 else 0.01 * x
*/
float leaky_reluf(float x) {
  return x > 0 ? x : .01f * x;
}

/*
  Mapping of Activation enum to proper activation function
    
  Parameters:
    x - Input value
    act - Activation function to apply
    
  Returns:
    Result of activation function
*/
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

/*
  Apply Softmax function to a Matrix
    
  Parameters:
    m - Input Matrix

  Note:
    Applies softmax to Matrix in place
    Will overwrite all values in Matrix
*/
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

/*
  Compute Derivative of Leaky ReLU
    
  Parameters:
    x - Input value
    
  Returns:
    Derivative of Leaky ReLU
*/
float deriv_leaky_reluf(float x) {
  return x > 0 ? 1.0f : .01f;
}

/*
  Compute Derivative of Activation Function

  Parameters:
    x - Input value
    act - Activation function

  Returns:
    Derivative of activation function
*/
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

/*
  Compute derivative of loss function

  Parameters:
    y_pred - Predicted value
    y_true - Real value
    loss - Loss function

  Returns:
    Derivative of loss function
*/
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

/*
  Process training data in batches
    
  Parameters:
    r - Memory region for temporary allocations
    b - Batch state tracking
    batch_size - Number of samples per batch
    n - Neural network to train
    m - Training data matrix
    lr - Learning rate
    config - Configuration of Network for Activations and Loss
*/
void batch_process(Region *r, Batch *b, size_t batch_size, NN n, Matrix m, float lr, NNConfig config) {
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
  // finite differences call, would not use
  // only really for testing and learning
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

/*
  Forward Pass through Attention Head

  Parameters:
    r - Memory region for tempory allocations
    mha - Attention Head parameter
    m - Input Matrix

  Returns:
    Output of attention
*/
Matrix attention_forward(Region *r, AttentionHead *mha, Matrix *m) {
  Matrix Q = matrix_alloc(r, m->rows, mha->Wq->cols);
  matrix_dot(Q, *m, *(mha->Wq));
  Matrix K = matrix_alloc(r, m->rows, mha->Wk->cols);
  matrix_dot(K, *m, *(mha->Wk));
  Matrix V = matrix_alloc(r, m->rows, mha->Wv->cols);
  matrix_dot(V, *m, *(mha->Wv));

  Matrix Q_split = split_heads(r, &Q, mha);
  Matrix K_split = split_heads(r, &K, mha);
  Matrix V_split = split_heads(r, &V, mha);

  Matrix attention = scaled_dot_product(r, &Q_split, &K_split, &V_split);

  Matrix concat = concat_heads(r, &attention, mha);
  Matrix output = matrix_alloc(r, concat.rows, mha->Wo->cols);
  matrix_dot(output, concat, *(mha->Wo));

  return output;
}

/*
  Forward Pass through feed-forward Network

  Parameters:
    r - Memory region for temporary allocations
    ff - Feed-Forward parameters
    m - Input Matrix
*/
Matrix feed_forward(Region *r, FeedForward *ff, Matrix *m) {
  Matrix hidden = matrix_alloc(r, m->rows, ff->W1->cols);
  matrix_dot(hidden, *m, *(ff->W1));
  add_bias(hidden, *ff->b1);
  matrix_act(hidden, RELU);

  Matrix output = matrix_alloc(r, hidden.rows, ff->W2->cols);
  matrix_dot(output, hidden, *(ff->W2));
  add_bias(output, *ff->b2);

  return output;
}

/*
  Apply Layer Normalization

  Parameters:
    r - Memory region for temporary allocations
    m - Input Matrix
    norm_p - Normalization Parameters

  Returns:
    Normalized Matrix
*/
Matrix layer_norm(Region *r, Matrix *m, Matrix *norm_p) {
  Matrix norm = matrix_alloc(r, m->rows, m->cols);
  for (size_t i = 0; i < m->rows; i++) {
    float sum = 0.0f;
    for (size_t j = 0; j < m->cols; j++) {
      sum += MAT_AT(*m, i, j);
    }

    float mean = sum / m->cols;
    float var = 0.0f;
    for (size_t j = 0; j < m->cols; j++) {
      float diff = MAT_AT(*m, i, j) - mean;
      var += diff * diff;
    }

    float std = sqrt(var / m->cols + 1e-5f);
    for (size_t j = 0; j < m->cols; j++) {
      float normal = (MAT_AT(*m, i, j) - mean) / std;
      MAT_AT(norm, i, j) = normal * MAT_AT(*norm_p, 0, j);
    }
  }

  return norm;
}

/*
  Forward Pass through Transformer layer

  Parameters:
    r - Memory region for temporary allocations
    tlayer - Transformer Layer parmeters
    m - Input Matrix

  Returns:
    Output of transformer layer
*/
Matrix tlayer_forward(Region *r, TransformerLayer *tlayer, Matrix *m) {
  Matrix attention = attention_forward(r, &tlayer->att, m);
  Matrix m_copy = matrix_alloc(r, m->rows, m->cols);
  matrix_copy(m_copy, *m);
  matrix_add(m_copy, attention);
  Matrix normal1 = layer_norm(r, &m_copy, tlayer->norm1);

  Matrix ffout = feed_forward(r, &tlayer->ff, &normal1);
  matrix_add(normal1, ffout);
  Matrix normal2 = layer_norm(r, &normal1, tlayer->norm2);

  return normal2;
}

/*
  Forward Pass through Transformer

  Parameters:
    r - Memory region for temporary allocations
    t - Transformer parameters
    m - Input Matrix

  Returns:
    Output of Transformer
*/
Matrix transformer_forward(Region *r, Transformer *t, Matrix *m) {
  Matrix out = *m;
  for (size_t i = 0; i < t->layers; i++) {
    out = tlayer_forward(r, &t->tlayers[i], &out);
  }

  if (t->encode != NULL) {
    matrix_dot(out, out, *t->encode);
  }

  return out;
}

/*
  Split matrix into multiple attention heads
    
  Parameters:
    r - Memory region for temporary allocations
    m - Input matrix
    mha - Attention Head parameters
    
  Returns:
    Matrix with Attention Heads concatenated along rows
*/
Matrix split_heads(Region *r, Matrix *m, AttentionHead *mha) {
  NN_ASSERT(m->cols % mha->att_heads == 0);
  size_t d = m->cols / mha->att_heads;

  Matrix result = matrix_alloc(r, m->rows * mha->att_heads, d);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < mha->att_heads; j++) {
      for (size_t k = 0; k < d; k++) {
        MAT_AT(result, i * mha->att_heads + j, k) = MAT_AT(*m, i, j * d + k);
      }
    }
  }

  return result;
}

/*
  Concatenate Attention Heads back into Matrix

  Parameters:
    r - Memory region for temporary allocations
    m - Input matrix
    mha - Attention Head parameters

  Returns:
    Reconstructed Matrix
*/
Matrix concat_heads(Region *r, Matrix *m, AttentionHead *mha) {
  NN_ASSERT(m->rows % mha->att_heads == 0);
  size_t original_rows = m->rows / mha->att_heads;

  Matrix result = matrix_alloc(r, original_rows, mha->att_heads * m->cols);
  for (size_t i = 0; i < original_rows; i++) {
    for (size_t j = 0; j < mha->att_heads; j++) {
      for (size_t k = 0; k < m->cols; k++) {
        MAT_AT(result, i, j * m->cols + k) = MAT_AT(*m, i * mha->att_heads + j, k);
      }
    }
  }

  return result;
}

/*
  Add bias Row Vector to a Matrix

  Parameters:
    m - Matrix
    b - Row Vector (bias)
*/
void add_bias(Matrix m, Row b) {
  NN_ASSERT(m.cols == b.cols);
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) += ROW_AT(b, j);
    }
  }
}

void softmax_backward(Matrix grad, Matrix probs) {
  for (size_t i = 0; i < grad.rows; i++) {
    float sum = 0.0f;
    for (size_t j = 0; j < grad.cols; j++) {
      float p = MAT_AT(probs, i, j);
      MAT_AT(grad, i, j) *= p * (1 - p);
      sum += MAT_AT(grad, i, j);
    }
    
    for (size_t j = 0; j < grad.cols; j++) {
      MAT_AT(grad, i, j) -= sum * MAT_AT(probs, i, j);
    }
  }
}

Transformer* transformer_alloc(Region *r, TConfig config) {
  Transformer *t = region_alloc(r, sizeof(Transformer));
  t->tlayers = region_alloc(r, sizeof(TransformerLayer) * config.layers);
  for (size_t i = 0; i < config.layers; i++) {
    t->tlayers[i] = *tlayer_alloc(r, config.arch[0], config.ff_arch[0], config.att_heads);
  }
  
  t->layers = config.layers;
  *t->encode = matrix_alloc(r, config.arch[0], config.arch[0]);
  t->arch = config.arch;

  return t;
}

TransformerLayer* tlayer_alloc(Region *r, size_t d_model, size_t d_ff, size_t heads) {
  TransformerLayer *tlayer = region_alloc(r, sizeof(TransformerLayer));
  *tlayer->att.Wq = matrix_alloc(r, d_model, d_model);
  *tlayer->att.Wk = matrix_alloc(r, d_model, d_model);
  *tlayer->att.Wv = matrix_alloc(r, d_model, d_model);
  *tlayer->att.Wo = matrix_alloc(r, d_model, d_model);
  tlayer->att.att_heads = heads;
  *tlayer->ff.W1 = matrix_alloc(r, d_model, d_ff);
  *tlayer->ff.W2 = matrix_alloc(r, d_ff, d_model);
  *tlayer->ff.b1 = row_alloc(r, d_model);
  *tlayer->ff.b2 = row_alloc(r, d_model);

  return tlayer;
}

void transformer_backprop(Region *r, Transformer *t, Transformer *grad_t, Matrix *in, Matrix *grad) {
  Matrix *curr_grad = grad;

  for (size_t i = t->layers - 1; i >= 0; i--) {
    TransformerLayer *tlayer = &(t->tlayers[i]);
    TransformerLayer *gradlayer = &(grad_t->tlayers[i]);

    Matrix *dnorm2 = norm_backward(r, curr_grad, tlayer->norm2_input, tlayer->norm2);
    matrix_copy(*gradlayer->norm2, *dnorm2);

    Matrix *dff = ff_backward(r, &(tlayer->ff), curr_grad, tlayer->ff_input, tlayer->ff_hidden);
    matrix_copy(*gradlayer->ff.W1, *dff);
    matrix_copy(*gradlayer->ff.W2, *dff);

    Matrix *dnorm1 = norm_backward(r, dff, tlayer->norm1_input, tlayer->norm1);
    matrix_copy(*gradlayer->norm1, *dnorm1);

    Matrix *datt = attention_backward(r, &tlayer->att, dnorm1, tlayer->att.Q, tlayer->att.K, 
      tlayer->att.V, tlayer->att.scores);
    matrix_copy(*gradlayer->att.Wq, *datt);
    matrix_copy(*gradlayer->att.Wk, *datt);
    matrix_copy(*gradlayer->att.Wv, *datt);
    matrix_copy(*gradlayer->att.Wo, *datt);

    curr_grad = datt;
  }
}

/*
  
*/
Matrix* norm_backward(Region *r, Matrix *grad, Matrix *x_forward, Matrix *gamma) {
  Matrix dx = matrix_alloc(r, grad->rows, grad->cols);
  Matrix d_gamma = matrix_alloc(r, 1, gamma->cols);

  for (size_t i = 0; i < x_forward->rows; i++) {
    float sum = 0.0f;
    for (size_t j = 0; j < x_forward->cols; j++) {
      sum += MAT_AT(*x_forward, i, j);
    }

    float mean = sum / x_forward->cols;
    float var = 0.0f;
    for (size_t j = 0; j < x_forward->cols; j++) {
      float diff = MAT_AT(*x_forward, i, j) - mean;
      var += diff * diff;
    }
    float std = sqrtf(var / x_forward->cols);

    float sum_dnorm = 0.0f;
    float sum_dnorm_diff = 0.0f;
    for (size_t j = 0; j < x_forward->cols; j++) {
      float diff = MAT_AT(*x_forward, i, j) - mean;
      float dnorm = MAT_AT(*grad, i, j) * MAT_AT(*gamma, 0, j);
      sum_dnorm += dnorm;
      sum_dnorm_diff += dnorm * diff;

      MAT_AT(d_gamma, i, j) += MAT_AT(*grad, i, j) * (diff / std);
    }

    for (size_t j = 0; j < x_forward->cols; j++) {
      float diff = MAT_AT(*x_forward, i, j) - mean;
      MAT_AT(dx, i, j) = MAT_AT(*grad, i, j) * 
        MAT_AT(*gamma, 0, j) - (sum_dnorm / x_forward->cols) -
        (diff * sum_dnorm_diff) / (x_forward->cols * std * std * std);
    }
  }

  return &dx;
}


Matrix* ff_backward(Region *r, FeedForward *ff, Matrix *grad, Matrix *input, Matrix *hidden) {
  Matrix dW2 = matrix_alloc(r, ff->W2->rows, ff->W2->cols);
  Matrix dW1 = matrix_alloc(r, ff->W1->rows, ff->W1->cols);
  Row db2 = row_alloc(r, ff->b2->cols);
  Row db1 = row_alloc(r, ff->b1->cols);
  Matrix dx = matrix_alloc(r, grad->rows, grad->cols);

  Matrix hidden_T = matrix_alloc(r, hidden->cols, hidden->rows);
  matrix_transpose(hidden_T, *hidden);
  matrix_dot(dW2, hidden_T, *grad);
  for (size_t j = 0; j < grad->cols; j++) {
    float sum = 0.0f;
    for (size_t i = 0; i < grad->rows; i++) {
      sum += MAT_AT(*grad, i, j);
    }

    ROW_AT(db2, j) = sum;
  }

    Matrix d_hidden = matrix_alloc(r, hidden->rows, hidden->cols);
    for (size_t i = 0; i < hidden->rows; i++) {
        for (size_t j = 0; j < hidden->cols; j++) {
            MAT_AT(d_hidden, i, j) = (MAT_AT(*hidden, i, j) > 0) ? MAT_AT(*grad, i, j) : 0.0f;
        }
    }
    
    Matrix input_T = matrix_alloc(r, input->cols, input->rows);
    matrix_transpose(input_T, *input);
    matrix_dot(dW1, input_T, d_hidden);
    
    for (size_t j = 0; j < d_hidden.cols; j++) {
        float sum = 0.0f;
        for (size_t i = 0; i < d_hidden.rows; i++) {
            sum += MAT_AT(d_hidden, i, j);
        }
        ROW_AT(db1, j) = sum;
    }
    
    Matrix W1_T = matrix_alloc(r, ff->W1->cols, ff->W1->rows);
    matrix_transpose(W1_T, *ff->W1);
    matrix_dot(dx, d_hidden, W1_T);
    
    matrix_copy(*ff->W1, dW1);
    matrix_copy(*ff->W2, dW2);
    row_copy(*ff->b1, db1);
    row_copy(*ff->b2, db2);
    
    return &dx;
}

Matrix* attention_backward(Region *r, AttentionHead *mha, Matrix *grad, Matrix *Q, Matrix *K, Matrix *V, Matrix *scores) {
  Matrix dx = matrix_alloc(r, grad->rows, grad->cols);
    
  Matrix dWo = matrix_alloc(r, mha->Wo->rows, mha->Wo->cols);
  Matrix grad_T = matrix_alloc(r, grad->cols, grad->rows);
  matrix_transpose(grad_T, *grad);
  matrix_dot(dWo, grad_T, *mha->Wo);
    
  Matrix dscores = matrix_alloc(r, scores->rows, scores->cols);
  Matrix Wo_T = matrix_alloc(r, mha->Wo->cols, mha->Wo->rows);
  matrix_transpose(Wo_T, *mha->Wo);
  matrix_dot(dscores, *grad, Wo_T);
    
  softmax_backward(dscores, *scores);
    
  float scale = 1.0f / sqrtf(mha->Wq->cols);
  matrix_scale(dscores, scale);
    
  Matrix dQ = matrix_alloc(r, Q->rows, Q->cols);
  Matrix dK = matrix_alloc(r, K->rows, K->cols);
  Matrix dV = matrix_alloc(r, V->rows, V->cols);
  
  Matrix K_T = matrix_alloc(r, K->cols, K->rows);
  matrix_transpose(K_T, *K);
  matrix_dot(dQ, dscores, K_T);
  Matrix Q_T = matrix_alloc(r, Q->cols, Q->rows);
  matrix_transpose(Q_T, *Q);
  matrix_dot(dK, Q_T, dscores);
  Matrix scores_T = matrix_alloc(r, scores->cols, scores->rows);
  matrix_transpose(scores_T, *scores);
  matrix_dot(dV, scores_T, *grad);
    
  matrix_add(dx, dQ);
  matrix_add(dx, dK);
  matrix_add(dx, dV);
    
  matrix_copy(*mha->Wq, dQ);
  matrix_copy(*mha->Wk, dK);
  matrix_copy(*mha->Wv, dV);
  matrix_copy(*mha->Wo, dWo);
    
  return &dx;
}

void transformer_learn() {
  
}

#endif // NN_IMPLEMENTATION
