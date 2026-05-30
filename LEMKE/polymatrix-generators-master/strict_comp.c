#include "strict_comp.h"
#include "coord_zero.h"

matrix_t **generate_strict_comp(int m, int n)
{
    matrix_t **game = generate_zero(m, n);

    double c1 = (double) rand() / (double) RAND_MAX;
    double c2 = (double) rand() / (double) RAND_MAX;
    double d1 = (double) rand() / (double) RAND_MAX;
    double d2 = (double) rand() / (double) RAND_MAX;
    
    matrix_mul_const_in(game[0], c1);
    matrix_mul_const_in(game[1], c2);
    matrix_add_const_in(game[0], d1);
    matrix_add_const_in(game[1], d2);

    return game;
}

polymatrix_t *generate_polymatrix_strict_comp(int actions, double p, char *graph, int g1, int g2)
{
    int players = graph_get_size(graph, g1, g2);
    polymatrix_t *game = polymatrix_alloc(players);
    graph_fill_char(game->graph, graph, g1, g2);

    int i, j;
    for (i = 0; i < players; ++i) {
        for (j = i; j < players; ++j) {
            if (game->graph[i][j] == 0)
                continue;
            matrix_t **bimatrix = generate_strict_comp(actions, actions);
            polymatrix_set_bimatrix(game, bimatrix[0], bimatrix[1], i, j);
            matrix_free(bimatrix[0]);
            matrix_free(bimatrix[1]);
            free(bimatrix);
        }
    }

    return game;
}
