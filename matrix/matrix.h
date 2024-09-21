#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>
#include <stdint.h>
#include "region.h"
#include <stdio.h>

#ifndef MAT_ASSERT
#include <assert.h>
#define MAT_ASSERT assert
#endif


typedef struct {
    size_t rows;
    size_t cols;
    float *elements;
} Matrix;

typedef struct {
    size_t cols;
    float *elements;
} Row;

Matrix matrix_alloc(Region *r, size_t rows, size_t cols);
void matrix_copy(Matrix destination, Matrix source);

#define MAT_AT(m, i , j) (m).elements[(i)*(m).cols + (j)]
#define MATRIX_PRINT(m) matrix_print(m, #m, 0);

void matrix_fill(Matrix m, float val);
void matrix_rand(Matrix m, float low, float high);
Row matrix_row(Matrix m, size_t row);
void matrix_print(Matrix m, const char *name, size_t padding);
void matrix_shuffle_rows(Matrix m);
bool matrices_equal(Matrix a, Matrix b);
int matrix_argmax(Matrix *m);
void matrix_save(Matrix *m, char* file_string);
Matrix *matrix_load(char* file_string);


#define row_alloc(r, cols) matrix_row(matrix_alloc(r, 1, cols), 0)
#define row_copy(destination, source) matrix_copy(row_as_matrix(destination), row_as_matrix(source))
#define row_rand(row, low, high) matrix_rand(row_as_matrix(row), low, high)
#define row_fill(row, x) matrix_fill(row_as_matrix(row), x)
#define row_print(row, name, padding) matrix_print(row_as_matrix(row), name, padding)
#define ROW_AT(row, col) (row).elements[col]

Matrix row_as_matrix(Row row);
Row row_slice(Row row, size_t i, size_t cols);

#endif // MATRIX_H_

#ifdef MATRIX_IMPLEMENTATION

Matrix matrix_alloc(Region *r, size_t rows, size_t cols) {
    if (rows < 1) rows = 1;
    if (cols < 1) cols = 1;

    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.elements = (float*) region_alloc(r, sizeof(*m.elements) * rows * cols);
    MAT_ASSERT(m.elements != NULL);
    return m;
}


Row matrix_row(Matrix m, size_t row) {
    return (Row) {
        .cols = m.cols,
        .elements = &MAT_AT(m, row, 0),
    };
}

void matrix_copy(Matrix destination, Matrix source) {
    if (matrices_equal(destination, source)) {
        return;
    }
    MAT_ASSERT(destination.rows == source.rows);
    MAT_ASSERT(destination.cols == source.cols);
    for (size_t i = 0; i < destination.rows; i++) {
        for (size_t j = 0; j < destination.cols; j++) {
            MAT_AT(destination, i, j) = MAT_AT(source, i, j);
        }
    }
}

void matrix_print(Matrix m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; i++) {
        printf("%*s   ", (int) padding, "");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s = ]\n", (int) padding, "");
}

void matrix_fill(Matrix m, float val) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i , j) = val;
        }
    }
}

float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

void matrix_rand(Matrix m, float low, float high) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void matrix_shuffle_rows(Matrix m) {
    for (size_t i = 0; i < m.rows; i++) {
        size_t j = i + rand() % (m.rows - i);
        if (i != j) {
            for (size_t k = 0; k < m.cols; k++) {
                float temp = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = temp;
            }
        }
    }
}

int matrix_argmax(Matrix *m) {
    int max_index = 0;
    float max_val = m->elements[0];

    for (size_t i = 0; i < m->rows * m->cols; i++) {
        if (m->elements[i] > max_val) {
            max_index = i;
            max_val = m->elements[i];
        }
    }

    return max_index;
}

void matrix_save(Matrix *m, const char *file_string) {
    FILE *file = fopen(file_string, "w");
    fprintf(file, "%d\n", m->rows);
    fprintf(file, "%d\n", m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            fprintf(file, "%.f\n", MAT_AT(*m, i, j));
        }
    }
    fclose(file);
    printf("Successfully saved matrix to %s\n", file_string);
}

Matrix *matrix_load(Region *r, const char *file_string) {
    FILE *file = fopen(file_string, "r");
    if (!file) {
        printf("Could not open file %s\n", file_string);
        return NULL;
    }

    char entry[256];
    if (!fgets(entry, sizeof(entry), file)) {
        printf("Failed to read rows from file %s\n", file_string);
        fclose(file);
        return NULL;
    }
    int rows = atoi(entry);

    if (!fgets(entry, sizeof(entry), file)) {
        printf("Failed to read cols from file %s\n", file_string);
        fclose(file);
        return NULL;
    }
    int cols = atoi(entry);

    Matrix *m = (Matrix *) region_alloc(r, sizeof(Matrix));
    if (!m) {
        printf("Failed to allocate memory for matrix\n");
        fclose(file);
        return NULL;
    }
    *m = matrix_alloc(r, rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (!fgets(entry, sizeof(entry), file)) {
                printf("Failed to read element at (%zu, %zu) from file %s\n", i , j, file_string);
                fclose(file);
                return NULL;
            }
            m->elements[i * cols + j] = atof(entry);
        }
    }

    printf("Successfully loaded matrix from file %s\n", file_string);
    fclose(file);
    return m;

}

Matrix row_as_matrix(Row row) {
    return (Matrix) {
        .rows = 1,
        .cols = row.cols,
        .elements = row.elements,
    };
}

Row row_slice(Row row, size_t i, size_t cols) {
    MAT_ASSERT(i < row.cols);
    MAT_ASSERT(i + cols <= row.cols);
    
    return (Row) {
        .cols = cols,
        .elements = &ROW_AT(row, i),
    };
}

#endif // MATRIX_IMPLEMENTATION