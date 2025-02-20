#pragma once

#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>
#include <stdint.h>
#include "../region/region.h"
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
//bool matrices_equal(Matrix a, Matrix b);
int matrix_argmax(Matrix *m);
void matrix_save(Matrix *m, const char *file_string);
Matrix *matrix_load(Region *r, const char *file_string);


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

/*
    Matrix allocation from memory region
    
    Parameters:
        r - Memory region for allocation
        rows - Number of rows (minimum 1)
        cols - Number of columns (minimum 1)
    
    Returns:
        Initialized Matrix structure with contiguous memory
        Elements are uninitialized by default
*/
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

/*
    Get row view of matrix
    
    Parameters:
        m - Source matrix
        row - Row index
    
    Returns:
        Row structure 
    
    Preconditions:
        row < m.rows
*/
Row matrix_row(Matrix m, size_t row) {
    return (Row) {
        .cols = m.cols,
        .elements = &MAT_AT(m, row, 0),
    };
}

/*
    Deep copy matrix contents
    
    Parameters:
        destination - Target matrix 
        source - Source matrix to copy from
    
    Preconditions:
        destination.rows == source.rows
        destination.cols == source.cols
*/
void matrix_copy(Matrix destination, Matrix source) {
    //if (matrices_equal(destination, source)) {
    //    return;
    //}
    MAT_ASSERT(destination.rows == source.rows);
    MAT_ASSERT(destination.cols == source.cols);
    for (size_t i = 0; i < destination.rows; i++) {
        for (size_t j = 0; j < destination.cols; j++) {
            MAT_AT(destination, i, j) = MAT_AT(source, i, j);
        }
    }
}

/*
    Print matrix contents with stdout
    
    Parameters:
        m - Matrix to print
        name - Label to display above matrix
        padding - Left-pad output with spaces
*/
void matrix_print(Matrix m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; i++) {
        printf("%*s   ", (int) padding, "");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s ]\n", (int) padding, "");
}

/*
    Fill matrix with given value
    
    Parameters:
        m - Matrix to modify
        val - Value to set all elements to
*/
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

/*
    Initialize matrix with random values
    
    Parameters:
        m - Matrix to initialize with random values
        low - Minimum value (inclusive)
        high - Maximum value (exclusive)
*/
void matrix_rand(Matrix m, float low, float high) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

/*
    Randomly shuffle rows of Matrix
    
    Parameters:
        m - Matrix to shuffle
    
    Notes:
        Implements Fisher-Yates shuffle
        Affects row order but preserves row contents
*/
void matrix_shuffle_rows(Matrix m) {
    for (size_t i = 0; i < m.rows; ++i) {
        size_t j = i + rand() % (m.rows - i);
        if (i != j) {
            for (size_t k = 0; k < m.cols; ++k) {
                float temp = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = temp;
            }
        }
    }
}

/*
    Find maximum value index
    
    Parameters:
        m - Input matrix
    
    Returns:
        Linear index of maximum element
        Returns first occurrence for multiple maxima
*/
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

/*
    Save matrix to file
    
    Parameters:
        m - Matrix to serialize
        file_string - Output file path
    
    File Format:
        First line: rows
        Second line: cols
        Subsequent lines: elements in row-major order
*/
void matrix_save(Matrix *m, const char *file_string) {
    FILE *file = fopen(file_string, "w");
    fprintf(file, "%zu\n", m->rows);
    fprintf(file, "%zu\n", m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            fprintf(file, "%.f\n", MAT_AT(*m, i, j));
        }
    }

    fclose(file);
    printf("Successfully saved matrix to %s\n", file_string);
}

/*
    Load matrix from file
    
    Parameters:
        r - Memory region for allocation
        file_string - Input file path
    
    Returns:
        Pointer to loaded matrix on success
        NULL on failure
    
    Notes:
        File format must match matrix_save()
        Allocates from region
*/
Matrix* matrix_load(Region *r, const char *file_string) {
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
    size_t rows = atoi(entry);

    if (!fgets(entry, sizeof(entry), file)) {
        printf("Failed to read cols from file %s\n", file_string);
        fclose(file);
        return NULL;
    }
    size_t cols = atoi(entry);

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

/*
    Convert Row to 1 row Matrix 
    
    Parameters:
        row - Row structure to convert
    
    Returns:
        Matrix structure 
*/
Matrix row_as_matrix(Row row) {
    return (Matrix) {
        .rows = 1,
        .cols = row.cols,
        .elements = row.elements,
    };
}

/*
    Create smaller, subsection of a Row
    
    Parameters:
        row - Source row
        i - Starting column index
        cols - Number of columns in slice
    
    Returns:
        New Row with portion of original
    
    Preconditions:
        i < row.cols
        i + cols <= row.cols
*/
Row row_slice(Row row, size_t i, size_t cols) {
    MAT_ASSERT(i < row.cols);
    MAT_ASSERT(i + cols <= row.cols);
    
    return (Row) {
        .cols = cols,
        .elements = &ROW_AT(row, i),
    };
}

#endif // MATRIX_IMPLEMENTATION
