#include "matrix.h"
#ifndef BIMATRIX_H
#define BIMATRIX_H

typedef struct bimatrix {
    matrix_t *R;
    matrix_t *C;
}bimatrix_t;

bimatrix_t *bimatrix_alloc(int m, int n);
void bimatrix_free(bimatrix_t *game);
void bimatrix_normalize(bimatrix_t *game);
void bimatrix_to_file(FILE *f, char *info, bimatrix_t *game);

#endif
