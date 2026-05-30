#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "coord_zero.h"
#include "graph.h"
#include "util.h"

matrix_t **generate_zero(int m, int n)
{
    matrix_t **game = malloc(sizeof(matrix_t *) * 2);

    game[0] = matrix_alloc(m, n);
    matrix_rand(game[0]);
    game[1] = matrix_mul_const(game[0], -1);

    return game;
}

matrix_t **generate_coord(int m, int n)
{
    matrix_t **game = malloc(sizeof(matrix_t *) * 2);

    game[0] = matrix_alloc(m, n);
    matrix_rand(game[0]);
    game[1] = matrix_copy(game[0]);

    return game;
}

//static matrix_t ** (*type[2])(int m, int n) = {generate_zero, generate_coord};

bayesian_t *generate_bayesian_zero_coord(int types, int actions, double p)
{
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);

    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            int k = (rand() < RAND_MAX * p) ? 0 : 1;
            matrix_t **bimatrix = (k) ? generate_zero(actions, actions) : generate_coord(actions, actions);
            bayesian_set_game(game, i, j, bimatrix);
        }
    }

    return game;
}

polymatrix_t *generate_polymatrix_zero_coord(int actions, double p, char *graph, int g1, int g2)
{
    int players = graph_get_size(graph, g1, g2);
    polymatrix_t *game = polymatrix_alloc(players);
    graph_fill_char(game->graph, graph, g1, g2);

    int i, j;
    int count = 0;
    for (i = 0; i < players; ++i) {
        for (j = i; j < players; ++j) {
            if (game->graph[i][j] == 0)
                continue;
            count++;
        }
    }
    int *coord = calloc(count, sizeof(int));
    for (i = 0; i < p * count; ++i) {
        coord[i] = 1;
    }

    int k = coord[myRandom(count)];
    for (i = 0; i < players; ++i) {
        for (j = i; j < players; ++j) {
            if (game->graph[i][j] == 0)
                continue;
            //int k = (rand() < RAND_MAX * p) ? 0 : 1;
            matrix_t **bimatrix = (k) ? generate_zero(actions, actions) : generate_coord(actions, actions);
            polymatrix_set_bimatrix(game, bimatrix[0], bimatrix[1], i, j);
            matrix_free(bimatrix[0]);
            matrix_free(bimatrix[1]);
            free(bimatrix);
            k = coord[myRandom(-1)];
        }
    }

    return game;
}

polymatrix_t *generate_polymatrix_group_zero(int actions, double p, char *graph, int g1, int g2)
{
    int players = graph_get_size(graph, g1, g2);
    polymatrix_t *game = polymatrix_alloc(players);
    graph_fill_char(game->graph, graph, g1, g2);

    int part = (int) p;

    int i, j;
    for (i = 0; i < players; ++i) {
        for (j = i; j < players; ++j) {
            if (game->graph[i][j] == 0)
                continue;
            int k = i % part == j % part;
            matrix_t **bimatrix = (!k) ? generate_zero(actions, actions) : generate_coord(actions, actions);
            polymatrix_set_bimatrix(game, bimatrix[0], bimatrix[1], i, j);
            matrix_free(bimatrix[0]);
            matrix_free(bimatrix[1]);
            free(bimatrix);
        }
    }

    return game;
}
