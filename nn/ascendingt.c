#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"

int main() {
    Region r = region_init(1024 * 1024 * 1024);

    // in this fake data, each sentence is a vector of 8 floats
    // there are 40 sentences
    size_t batch_size = 40;
    size_t d_model = 8;
    Matrix toy_data = matrix_alloc(&r, batch_size, d_model);
    for (size_t i = 0; i < toy_data.rows; i++) {
        for (size_t j = 0; j < toy_data.cols; j++) {
            MAT_AT(toy_data, i, j) = (float)(i + j) / 10.0f;
        }
    }
    matrix_print(toy_data, "Toy Data", 4);

    // simple transformer to test on fake data
    Transformer transformer;
    transformer.layers = 3;
    transformer.tlayers = malloc(sizeof(TransformerLayer) * transformer.layers);

    for (size_t i = 0; i < transformer.layers; i++) {
        TransformerLayer *layer = &transformer.tlayers[i];
        layer->att.att_heads = 1;

        layer->att.Wq = malloc(sizeof(Matrix));
        layer->att.Wk = malloc(sizeof(Matrix));
        layer->att.Wv = malloc(sizeof(Matrix));
        layer->att.Wo = malloc(sizeof(Matrix));
        *layer->att.Wq = matrix_alloc(&r, d_model, d_model);
        *layer->att.Wk = matrix_alloc(&r, d_model, d_model);
        *layer->att.Wv = matrix_alloc(&r, d_model, d_model);
        *layer->att.Wo = matrix_alloc(&r, d_model, d_model);

        matrix_rand(*layer->att.Wq, -0.1f, 0.1f);
        matrix_rand(*layer->att.Wk, -0.1f, 0.1f);
        matrix_rand(*layer->att.Wv, -0.1f, 0.1f);
        matrix_rand(*layer->att.Wo, -0.1f, 0.1f);

        layer->ff.W1 = malloc(sizeof(Matrix));
        layer->ff.W2 = malloc(sizeof(Matrix));
        layer->ff.b1 = malloc(sizeof(Matrix));
        layer->ff.b2 = malloc(sizeof(Matrix));
        *layer->ff.W1 = matrix_alloc(&r, d_model, 16);
        *layer->ff.W2 = matrix_alloc(&r, 16, d_model);
        *layer->ff.b1 = row_alloc(&r, 16);
        *layer->ff.b2 = row_alloc(&r, d_model);

        matrix_rand(*layer->ff.W1, -0.1f, 0.1f);
        matrix_rand(*layer->ff.W2, -0.1f, 0.1f);
        row_rand(*layer->ff.b1, -0.1f, 0.1f);
        row_rand(*layer->ff.b2, -0.1f, 0.1f);

        layer->norm1 = malloc(sizeof(Matrix));
        layer->norm2 = malloc(sizeof(Matrix));
        *layer->norm1 = matrix_alloc(&r, 1, d_model);
        *layer->norm2 = matrix_alloc(&r, 1, d_model);

        for (size_t j = 0; j < d_model; j++) {
            MAT_AT(*(layer->norm1), 0, j) = 1.0f;
            MAT_AT(*(layer->norm2), 0, j) = 1.0f;
        }
    }

    // No encoder for basic testing
    transformer.encode = NULL;

    Matrix output = transformer_forward(&r, &transformer, &toy_data);
    matrix_print(output, "Transformer Output", 4);

    region_reset(&r);
    free(transformer.tlayers);

    return 0;
}