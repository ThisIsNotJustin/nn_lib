#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <libkern/OSByteOrder.h>
#include <string.h>
#include <time.h>

#include "nn.h"

#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#define MNIST_IMG_SIZE 784 // 28x28
#define MNIST_LABEL_SIZE 10 // 0-9
#define TRAIN_IMAGE "/Users/justin/Documents/Projects/nn_library/mnist/train-images.idx3-ubyte"
#define TRAIN_LABEL "/Users/justin/Documents/Projects/nn_library/mnist/train-labels.idx1-ubyte"
#define TEST_IMAGE "/Users/justin/Documents/Projects/nn_library/mnist/t10k-images.idx3-ubyte"
#define TEST_LABEL "/Users/justin/Documents/Projects/nn_library/mnist/t10k-labels.idx1-ubyte"

void FlipLong(unsigned char* ptr) {
  register unsigned char temp;

  temp = *(ptr);
  *(ptr) = *(ptr + 3);
  *(ptr + 3) = temp;

  ptr += 1;
  temp = *(ptr);
  *(ptr) = *(ptr + 1);
  *(ptr + 1) = temp;
}

int read_mnist_images(const char* file_path, Matrix data, int num_samples) {
    int fd, i, j;
    unsigned char pixel;
    unsigned char *ptr;
    int info[4];  // MNIST image file header is 4 integers

    if ((fd = open(file_path, O_RDONLY)) == -1) {
        printf("Error opening MNIST image file\n");
        return -1;
    }

    read(fd, info, 4 * sizeof(int));
    for (i = 0; i < 4; ++i) {
      ptr = (unsigned char*)(info + i);
      FlipLong(ptr);
      ptr = ptr + sizeof(int);
    }

    // Check file header
    if (info[0] != 2051 || info[1] != num_samples || info[2] != 28 || info[3] != 28) {
        printf("Unexpected image file header\n");
        close(fd);
        return -1;
    }

    for (i = 0; i < num_samples; ++i) {
        for (j = 0; j < MNIST_IMG_SIZE; ++j) {
            if (read(fd, &pixel, 1) != 1) {
                printf("Error reading image data\n");
                close(fd);
                return -1;
            }
            MAT_AT(data, i, j) = (float)pixel / 255.0f;
        }
    }

  close(fd);
  return 0;
}

int read_mnist_labels(const char* file_path, Matrix data, int num_samples) {
    int fd, i, k;
    unsigned char label;
    unsigned char *ptr;
    int info[2];  // MNIST label file header is 2 integers

    if ((fd = open(file_path, O_RDONLY)) == -1) {
        printf("Error opening MNIST label file\n");
        return -1;
    }

    read(fd, info, 2 * sizeof(int));
    for (i = 0; i < 2; ++i) {
      ptr = (unsigned char*) (info + i);
      FlipLong(ptr);
      ptr = ptr +sizeof(int);
    }

    // Check file header
    if (info[0] != 2049 || info[1] != num_samples) {
        printf("Unexpected label file header\n");
        close(fd);
        return -1;
    }

    for (i = 0; i < num_samples; ++i) {
        if (read(fd, &label, 1) != 1) {
            printf("Error reading label data\n");
            close(fd);
            return -1;
        }
        for (k = 0; k < MNIST_LABEL_SIZE; ++k) {
            MAT_AT(data, i, MNIST_IMG_SIZE + k) = (k == label) ? 1.0f : 0.0f;
        }
    }

    close(fd);
    return 0;
}

Matrix load_mnist(Region* r, const char* img_file, const char* label_file, int num_samples) {
    Matrix data = matrix_alloc(r, num_samples, MNIST_IMG_SIZE + MNIST_LABEL_SIZE);
    
    if (read_mnist_images(img_file, data, num_samples) != 0 ||
        read_mnist_labels(label_file, data, num_samples) != 0) {
        printf("Error reading MNIST Data\n");
        exit(1);
    }

    return data;
}

int main() {
  srand(time(0));

  Region perm = region_init(1024 * 1024 * 1024);
  Region temp = region_init(1024 * 1024 * 1024);

  size_t arch[] = {784, 64, 32, MNIST_LABEL_SIZE};
  size_t arch_count = sizeof(arch) / sizeof(arch[0]);

  printf("Network architecture:\n");
  for (size_t i = 0; i < arch_count; i++) {
    printf("Layer %zu: %zu neurons\n", i, arch[i]);
  }

  NN n = nn_alloc(&perm, arch, arch_count);
  nn_rand(n, -0.1, 0.1);

  printf("Network input size: %zu\n", NN_INPUT(n).cols);
  printf("Network output size: %zu\n", NN_OUTPUT(n).cols);

  if (NN_INPUT(n).cols != MNIST_IMG_SIZE || NN_OUTPUT(n).cols != MNIST_LABEL_SIZE) {
    printf("Error: Network architecture does not match MNIST data dimensions\n");
    exit(1);
  }

  printf("Loading training data\n");
  Matrix train_data = load_mnist(&perm, TRAIN_IMAGE, TRAIN_LABEL, 60000);
  printf("Loaded %zu training samples\n", train_data.rows);

  float lr = 0.01f;
  size_t epochs = 50;
  Batch batch = {0};
  NNConfig config = {
    .act = SOFTMAX,
    .loss = CCE
  };

  printf("Training the network\n");
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    while (!batch.finished) {
      batch_process(&temp, &batch, 32, n, train_data, lr, config);
    }
    region_reset(&temp);

    printf("Epoch %zu completed. Average cost: %f\n", epoch + 1, batch.cost);
  }

  printf("Loading test data\n");
  Matrix test_data = load_mnist(&perm, TEST_IMAGE, TEST_LABEL, 10000);

  printf("Testing the network\n");
  size_t correct = 0;

  for (size_t i = 0; i < test_data.rows; i++) {
    Row test_row = matrix_row(test_data, i);
    
    Row input = NN_INPUT(n);
    Row image_part = row_slice(test_row, 0, MNIST_IMG_SIZE);
    row_copy(input, image_part);
    
    nn_forward(n, config.act);
    
    Row output_row = NN_OUTPUT(n);
    size_t predicted = 0;
    float max = ROW_AT(output_row, 0);
    for (size_t k = 1; k < MNIST_LABEL_SIZE; k++) {
        if (ROW_AT(output_row, k) > max) {
            max = ROW_AT(output_row, k);
            predicted = k;
        }
    }
    
    size_t actual = 0;
    for (size_t k = 0; k < MNIST_LABEL_SIZE; k++) {
        if (ROW_AT(test_row, MNIST_IMG_SIZE + k) == 1.0f) {
            actual = k;
            break;
        }
    }
    
    printf("Predicted: %zu, Actual: %zu\n", predicted, actual);
    if (predicted == actual) {
        correct++;
    }
    
    region_reset(&temp);
  }

  float accuracy = (float) correct / test_data.rows * 100.0f;
  printf("Test Accuracy: %.2f%%\n", accuracy);

  Row first_image = matrix_row(test_data, 0);
  for (size_t i = 0; i < MNIST_IMG_SIZE; i++) {
      printf("%f ", ROW_AT(first_image, i));
  }
  printf("\n");

  Row first_label = matrix_row(test_data, 0);
  for (size_t i = MNIST_IMG_SIZE; i < MNIST_IMG_SIZE + MNIST_LABEL_SIZE; i++) {
      if (ROW_AT(first_label, i) == 1.0) {
          printf("Corresponding digit: %zu\n", i - MNIST_IMG_SIZE);
          break;
      }
  }
  printf("\n");

  region_reset(&perm);
  region_reset(&temp);

  return 0;
}
