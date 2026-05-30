#include <stdio.h>
#include <stdlib.h>
#include "polymatrix.h"

polymatrix_t *polymatrix_alloc(int players)
{
    int i;
    polymatrix_t *game = malloc(sizeof(polymatrix_t));

    game->players = players;

    game->graph = malloc(sizeof(int *) * players);
    for (i = 0; i < players; i++)
        game->graph[i] = calloc(players, sizeof(int));

    game->payoffs = malloc(sizeof(matrix_t **) * players);
    int j;
    for (i = 0; i < players; i++) {
        game->payoffs[i] = malloc(sizeof(matrix_t *) * players);
        for (j = 0; j < players; j++)
            game->payoffs[i][j] = NULL;
    }

    return game;
}

void polymatrix_free(polymatrix_t *game)
{
    int i, j;

    for (i = 0; i < game->players; ++i) {
        for (j = 0; j < game->players; ++j)
            if (game->payoffs[i][j])
                matrix_free(game->payoffs[i][j]);
        free(game->graph[i]);
        free(game->payoffs[i]);
    }

    free(game->graph);
    free(game->payoffs);
    free(game);
}

int count_strategies(polymatrix_t *game, int *s_count)
{
    int count = 0;
    int i, j;
    
    for (i = 0; i < game->players; ++i)
        for (j = 0; j < game->players; ++j)
            if (game->payoffs[i][j]){
                s_count[i] = game->payoffs[i][j]->nrows;
                count += s_count[i];
                break;
            }

    return count;
}

void polymatrix_set_bimatrix(polymatrix_t *game, matrix_t *R, matrix_t *C, int i, int j)
{
    game->payoffs[i][j] = matrix_copy(R);
    game->payoffs[j][i] = matrix_trans(C);
}

void polymatrix_upper_lower_i_row(polymatrix_t *game, int i, int r, double *U, double *L)
{
    int j, k;
    double m, M;
    *U = 0;
    *L = 0;

    for (j = 0; j < game->players; ++j) {
        if (!game->payoffs[i][j])
            continue;
        matrix_t *A = game->payoffs[i][j];

        m = M = A->data[r][0];
        for (k = 1; k < A->ncols; ++k) {
            if (m > A->data[r][k])
                m = A->data[r][k];
            if (M < A->data[r][k])
                M = A->data[r][k];
        }
        *U += M;
        *L += m;
    }
}

void polymatrix_upper_lower_i(polymatrix_t *game, int i, int n, double *U, double *L)
{
    int p;
    double m, M;
    polymatrix_upper_lower_i_row(game, i, 0, U, L);

    for (p = 1; p < n; ++p) {
        polymatrix_upper_lower_i_row(game, i, p, &M, &m);
        if (M > *U)
            *U = M;
        if (m < *L)
            *L = m;
    }
}

void polymatrix_normalize(polymatrix_t *game, int *count)
{
    int i, j;
    double U, L;
    for (i = 0; i < game->players; ++i) {
        polymatrix_upper_lower_i(game, i, count[i], &U, &L);
        int di = 0;
        for (j = 0; j < game->players; ++j) {
            if (!game->payoffs[i][j])
                continue;
            di++;
        }
        for (j = 0; j < game->players; ++j) {
            if (!game->payoffs[i][j])
                continue;
            matrix_add_const_in(game->payoffs[i][j], (-L / di));
            matrix_mul_const_in(game->payoffs[i][j], 1 / (U - L));
        }
    }
}

void polymatrix_to_file(FILE *f, char *info, polymatrix_t *game)
{
    int i, j;

    fprintf(f, "%d\n", game->players);

    for (i = 0; i < game->players; i++)
        for (j = 0; j < game->players; j++)
            fprintf(f, "%d ", game->graph[i][j]);

    for (i = 0; i < game->players; i++)
        for (j = i + 1; j < game->players; j++) {
            if (game->payoffs[i][j] != NULL) {
                fprintf(f, "%d %d ", game->payoffs[i][j]->nrows, game->payoffs[i][j]->ncols);
                matrix_write(game->payoffs[i][j], f);
                matrix_write(game->payoffs[j][i], f);
            }
        }
}

void polymatrix_print(polymatrix_t *game)
{
    int i, j;
    printf("Graph\n");
    for (i = 0; i < game->players; i++){
        for (j = 0; j < game->players; j++) {
            printf("%d ", game->graph[i][j]);
        }
        printf("\n");
    }

    printf("Games\n");
    for (i = 0; i < game->players; i++) {
        for (j = 0; j < game->players; j++) {
            if (game->payoffs[i][j] != NULL) {
                printf("Player %d v %d\n", i, j);
                printf("%d %d\n", game->payoffs[i][j]->nrows, game->payoffs[i][j]->ncols);
                matrix_print(game->payoffs[i][j]);
            }
        }
    }
}

int polymatrix_size(polymatrix_t *game)
{
    int i, j, count;
    count = 0;
    for (i = 0; i < game->players; ++i) {
        for (j = 0; j < game->players; ++j) {
            if (game->payoffs[i][j] != NULL)
                count += game->payoffs[i][j]->nrows * game->payoffs[i][j]->ncols;
        }
    }
    return count;
}
