#include "weight_coop.h"
#include "util.h"

matrix_t **generate_weight_coop(int m, double n, int *a1, int *a2)
{
    matrix_t **game = malloc(sizeof(matrix_t *) * 2);

    game[0] = matrix_alloc(m, m);
    game[1] = matrix_alloc(m, m);

    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < m; ++j) {
            if (a1[i] == a2[j]) {
                game[0]->data[i][j] = n;
                game[1]->data[i][j] = n;
            }
        }
    }

    return game;
}

polymatrix_t *generate_polymatrix_weight_coop(int actions, double p, char *graph, int g1, int g2)
{
    int players = graph_get_size(graph, g1, g2);
    polymatrix_t *game = polymatrix_alloc(players);
    graph_fill_char(game->graph, graph, g1, g2);

    int **acts = malloc(sizeof(int *) * players);
    int i, j;
    for (i = 0; i < players; ++i) {
        acts[i] = malloc(sizeof(int) * actions);
        acts[i][0] = myRandom(actions * p);
        for (j = 1; j < actions; ++j) {
            acts[i][j] = myRandom(-1);
        }
    }

    for (i = 0; i < players; ++i) {
        for (j = i; j < players; ++j) {
            if (game->graph[i][j] == 0)
                continue;
            double tmp = (((double) rand() / (double) RAND_MAX) * 10);
            matrix_t **bimatrix = generate_weight_coop(actions, tmp, acts[i], acts[j]);
            polymatrix_set_bimatrix(game, bimatrix[0], bimatrix[1], i, j);
            matrix_free(bimatrix[0]);
            matrix_free(bimatrix[1]);
            free(bimatrix);
        }
    }

    return game;
}
