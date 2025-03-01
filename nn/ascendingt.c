#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"

int main() {
    Region r = region_init(1024 * 1024 * 1024);

    size_t arch[] = {8};
    size_t ff_arch[] = {16};

    TConfig config = {
        .layers = 3,
        .arch = arch,
        .att_heads = 1,
        .ff_arch = ff_arch
    };

    Transformer *t = transformer_alloc(&r, config);
    Transformer *grad_t = transformer_alloc(&r, config);

    for (size_t i = 0; i < t->layers; i++) {
        matrix_rand(*t->tlayers[i].att.Wq, -0.1, 0.1);
        matrix_rand(*t->tlayers[i].att.Wk, -0.1, 0.1);
        matrix_rand(*t->tlayers[i].att.Wv, -0.1, 0.1);
        matrix_rand(*t->tlayers[i].att.Wo, -0.1, 0.1);
        
        matrix_rand(*t->tlayers[i].ff.W1, -0.1, 0.1);
        matrix_rand(*t->tlayers[i].ff.W2, -0.1, 0.1);
        row_fill(*t->tlayers[i].ff.b1, 0);
        row_fill(*t->tlayers[i].ff.b2, 0);

        matrix_fill(*t->tlayers[i].norm1, 1.0f);
        matrix_fill(*t->tlayers[i].norm2, 1.0f);
    }

    // in this fake data, each sentence is a vector of 8 floats
    // there are 40 sentences
    size_t batch_size = 40;
    size_t d_model = 8;
    Matrix *toy_data = matrix_alloc(&r, batch_size, d_model);
    for (size_t i = 0; i < toy_data->rows; i++) {
        for (size_t j = 0; j < toy_data->cols; j++) {
            MAT_AT(*toy_data, i, j) = (float)(i + j) / 10.0f;
        }
    }
    matrix_print(*toy_data, "Toy Data", 4);

    // Create a gradient matrix (example)
    Matrix *grad_output = matrix_alloc(&r, batch_size, d_model);
    for (size_t i = 0; i < grad_output->rows; i++) {
        for (size_t j = 0; j < grad_output->cols; j++) {
            MAT_AT(*grad_output, i, j) = 1.0f;
        }
    }

    // Forward pass
    Matrix *output = transformer_forward(&r, t, toy_data);
    matrix_print(*output, "Transformer Output", 4);

    // Backpropagation
    transformer_backprop(&r, t, grad_t, toy_data, grad_output);

    // Learning
    float learning_rate = 0.01f;
    transformer_learn(t, grad_t, learning_rate);

    // Print updated weights (example)
    matrix_print(*t->tlayers[0].att.Wq, "Updated Wq", 4);

    region_reset(&r);

    return 0;
}