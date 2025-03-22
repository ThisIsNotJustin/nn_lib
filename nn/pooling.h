#ifndef POOLING_H
#define POOLING_H

#include "../matrix/matrix.h"
#include "../region/region.h"

typedef struct {
    size_t pool_size;
    size_t stride;
    Matrix *input;
    Matrix *output;
    Matrix *mask;
} MaxPoolingLayer;

MaxPoolingLayer* pooling_layer_alloc(Region *r, size_t pool_size, size_t stride);
Matrix* pooling_forward(Region *r, MaxPoolingLayer *pl, Matrix *input);
Matrix* pooling_backward(Region *r, MaxPoolingLayer *pl, Matrix *output);

#endif // POOLING_H