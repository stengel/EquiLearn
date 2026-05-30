#include <stdlib.h>
#include <stdio.h>
#include "bimatrix.h"

static int prec = 15;

bimatrix_t *bimatrix_alloc(int m, int n)
{
    bimatrix_t *game = malloc(sizeof(bimatrix_t));
    game->R = matrix_alloc(m, n);
    game->C = matrix_alloc(m, n);
    return game;
}

void bimatrix_free(bimatrix_t *game)
{
    matrix_free(game->R);
    matrix_free(game->C);
    free(game);
}

void bimatrix_normalize(bimatrix_t *game)
{
    matrix_t *m;
    m = matrix_norm(game->R);
    matrix_free(game->R);
    game->R = m;

    m = matrix_norm(game->C);
    matrix_free(game->C);
    game->C = m;
}

void bimatrix_to_file(FILE *f, char *info, bimatrix_t *game)
{
    int i, j;
    double a, b;

    fprintf(f, "NFG 1 D \"Using bm-gen\nGame Info\n");
    fprintf(f, "%s ", info);

    fprintf(f, "{ \"P1\" \"P2\"}");

    int m = game->R->nrows;
    int n = game->R->ncols;
    fprintf(f,"{ %d %d }\n\n", m, n);

    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            a =(game->R->data[j][i]);
            b =(game->C->data[j][i]);
            fprintf(f,"%.*lf %.*lf ", prec, a, prec, b);
        }
    }
}
