#ifndef CONV2D_H
#define CONV2D_H

#include "../matrix/matrix.h"
#include "../region/region.h"

typedef enum {
    VALID,
    SAME
} PaddingType;

typedef struct {
    Matrix *filters;
    Matrix *d_filters;

    Matrix *bias;
    Matrix *d_bias;

    size_t kernel_size;
    size_t num_filters;
    size_t stride;
    PaddingType padding;

    Matrix *input;
    Matrix *output;
} Conv2DLayer;

Conv2DLayer* conv_layer_alloc(Region *r, size_t kernel_size, size_t num_filters, size_t stride, PaddingType padding, size_t input_channels);
Matrix* conv_forward(Region *r, Conv2DLayer *layer, Matrix *input);
Matrix* conv_backward(Region *r, Conv2DLayer *layer, Matrix *output);
void conv_learn(Conv2DLayer *layer, float lr);

#endif // CONV2d_H