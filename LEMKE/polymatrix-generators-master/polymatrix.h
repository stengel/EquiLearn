#include "matrix.h"
#ifndef POLYMATRIX_H
#define POLYMATRIX_H

typedef struct polymatrix {
    int players;
    int **graph;
    matrix_t ***payoffs;
}polymatrix_t;

polymatrix_t *polymatrix_alloc(int players);
void polymatrix_free(polymatrix_t *game);
void polymatrix_set_bimatrix(polymatrix_t *game, matrix_t *A, matrix_t *B, int i, int j);
void polymatrix_normalize(polymatrix_t *game, int *count);
void polymatrix_to_file(FILE *f, char *info, polymatrix_t *game);
void polymatrix_print(polymatrix_t *game);
int polymatrix_size(polymatrix_t *game);

#endif
