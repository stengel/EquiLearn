#include <math.h>
#include <limits.h>
#include "matrix.h"
#include "bayesian.h"
#include "graph.h"

int dist_type;

bayesian_t *bayesian_alloc(int m, int n)
{
    int i;
    bayesian_t *game = malloc(sizeof(bayesian_t));

    game->m = m;
    game->n = n;

    game->distribution = matrix_alloc(m, n);
    game->payoffs = malloc(sizeof(matrix_t **) * (m + n));
    for (i = 0; i < m; i++)
        game->payoffs[i] = malloc(sizeof(matrix_t *) * n);
    for (i = 0; i < n; i++)
        game->payoffs[i+m] = malloc(sizeof(matrix_t *) * m);

    return game;
}

void bayesian_free(bayesian_t *game)
{
    matrix_free(game->distribution);

    int i, j;
    for (i = 0; i < game->m; ++i) {
        for (j = 0; j < game->n; ++j) 
            matrix_free(game->payoffs[i][j]);
        free(game->payoffs[i]);
    }

    for (i = 0; i < game->n; ++i) {
        for (j = 0; j < game->m; ++j)
            matrix_free(game->payoffs[i+game->m][j]);
        free(game->payoffs[i+game->m]);
    }

    free(game->payoffs);
    free(game);
}

polymatrix_t *bayesian_to_polymatrix(bayesian_t *game)
{
    polymatrix_t *poly = polymatrix_alloc(game->m + game->n);

    graph_fill_complete_bipartite(poly->graph, game->m, game->n);
    int i, j;
    double pi, pi_j;
    for (i = 0; i < game->m; i++) {
        pi = 0;
        for (j = 0; j < game->n; j++)
            pi += game->distribution->data[i][j];
        for (j = 0; j < game->n; j++) {
            //printf("Adding %d %d\n", i, j + game->m);
            pi_j = game->distribution->data[i][j] / pi;
            poly->payoffs[i][j + game->m] = matrix_mul_const(game->payoffs[i][j], pi_j);
        }
    }

    for (i = 0; i < game->n; i++) {
        pi = 0;
        for (j = 0; j < game->n; j++)
            pi += game->distribution->data[j][i];
        for (j = 0; j < game->m; j++) {
            pi_j = game->distribution->data[j][i] / pi;
            poly->payoffs[i+game->m][j] = matrix_mul_const(game->payoffs[i + game->m][j], pi_j);
        }
    }

    return poly;
}

void bayesian_random(bayesian_t *game, int s)
{
    int i, j;
    for (i = 0; i < game->m; i++)
        for (j = 0; j < game->n; j++){
            game->payoffs[i][j] = matrix_alloc(s, s);
            matrix_rand(game->payoffs[i][j]);
        }

    for (i = 0; i < game->n; i++)
        for (j = 0; j < game->m; j++) {
            game->payoffs[i + game->m][j] = matrix_alloc(s, s);
            matrix_rand(game->payoffs[i + game->m][j]);
        }
}
void bayesian_fill_distribution_uniform(bayesian_t *game)
{
    int i, j;

    for (i = 0; i < game->m; i++)
        for (j = 0; j < game->n; j++)
            game->distribution->data[i][j] = 1.0 / (double)(game->m * game->n);
}

void bayesian_fill_distribution_rand(bayesian_t *game)
{
    int i, j;
    double sum = 0;

    for (i = 0; i < game->m; ++i)
        for (j = 0; j < game->n; j++) {
            double k = (double)rand() / RAND_MAX;
            sum += k;
            game->distribution->data[i][j] = k;
        }

    for (i = 0; i < game->m; ++i)
        for (j = 0; j < game->n; j++)
            game->distribution->data[i][j] /= sum;
}

void bayesian_fill_distribution(bayesian_t *game)
{
    if (dist_type == 0)
        bayesian_fill_distribution_uniform(game);
    else
        bayesian_fill_distribution_rand(game);
}


void bayesian_fill_game(bayesian_t *game, int s)
{
    bayesian_fill_distribution(game);
    bayesian_random(game, s);
}

bayesian_t *generate_random_bayesian(int types, int actions)
{
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);
    bayesian_random(game, actions);

    return game;
}

void bayesian_print(bayesian_t *game)
{
    printf("Distributions\n");
    matrix_print(game->distribution);
    printf("Payoffs\n");
    int i, j;

    printf("L->R\n");
    for (i = 0; i < game->m; i++) {
        for (j = 0; j < game->n; j++){
            printf("Types %d %d\n", i, j);
            matrix_print(game->payoffs[i][j]);
        }
    }

    printf("R->L\n");
    for (i = 0; i < game->n; i++) 
        for (j = 0; j < game->m; j++){
            printf("Types %d %d\n", i, j);
            matrix_print(game->payoffs[i + game->m][j]);
        }
}

void bayesian_set_game(bayesian_t *game, int i, int j, matrix_t **bimat)
{
    game->payoffs[i][j] = matrix_copy(bimat[0]);
    game->payoffs[j+game->m][i] = matrix_trans(bimat[1]);
    matrix_free(bimat[0]);
    matrix_free(bimat[1]);
    free(bimat);
}
