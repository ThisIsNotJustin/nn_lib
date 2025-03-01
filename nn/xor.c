#include <time.h>
#include <stdio.h>

#define NN_IMPLEMENTATION
#include "nn.h"

int main(void) {
    Region r = region_init(1024 * 1024 * 100);
    
    float training_data[] = {
        0, 0,    0,
        0, 1,    1,
        1, 0,    1,
        1, 1,    0
    };

    Matrix m = *matrix_alloc(&r, 4, 3);
    for (size_t i = 0; i < 4; i++) {
        MAT_AT(m, i, 0) = training_data[i * 3];
        MAT_AT(m, i, 1) = training_data[i * 3 + 1];
        MAT_AT(m, i, 2) = training_data[i * 3 + 2];
    }

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(&r, arch, 3);
    NNConfig config = {
        .loss = BCE,
        .act = SIG
    };
    nn_rand(nn, -0.5, 0.5);
    
    float learning_rate = 0.1f;
    size_t epochs = 30000;
    size_t batch_size = 2;
    
    // Training
    Batch batch = {0};
    for (size_t i = 0; i < epochs; i++) {
        batch_process(&r, &batch, batch_size, nn, m, learning_rate, config);
        if (batch.finished && i % 100 == 0) {
            printf("Epoch %zu: cost = %f\n", i, batch.cost);
        }
    }

    // Testing
    printf("\nTesting XOR:\n");
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            ROW_AT(NN_INPUT(nn), 0) = i;
            ROW_AT(NN_INPUT(nn), 1) = j;
            nn_forward(nn, config.act);
            printf("%zu XOR %zu = %f\n", i, j, ROW_AT(NN_OUTPUT(nn), 0));
        }
    }

    return 0;
}