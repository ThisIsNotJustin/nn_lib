CC = gcc
CFLAGS = -Wall -Wextra -O2 -I.
DEFINES = -DNN_IMPLEMENTATION -DREGION_IMPLEMENTATION -DMATRIX_IMPLEMENTATION -DLA_IMPLEMENTATION

# Source files
XOR_SRC = nn/xor.c
MNIST_SRC = nn/mnist.c

# Build directory and targets
BUILD_DIR = build
XOR_TARGET = $(BUILD_DIR)/xor
MNIST_TARGET = $(BUILD_DIR)/mnist

.PHONY: all clean xor mnist

all: xor mnist

xor: $(XOR_TARGET)

mnist: $(MNIST_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(XOR_TARGET): $(XOR_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ $<

$(MNIST_TARGET): $(MNIST_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)