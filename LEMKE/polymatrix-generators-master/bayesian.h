#include "matrix.h"
#include "polymatrix.h"
#ifndef BAYESIAN_H
#define BAYESIAN_H

extern int dist_type;
typedef struct bayesian{
    int m;
    int n;
    matrix_t *distribution;
    matrix_t ***payoffs;
}bayesian_t;

bayesian_t *bayesian_alloc(int t1, int t2);
void bayesian_free(bayesian_t *game);
polymatrix_t *bayesian_to_polymatrix(bayesian_t *game);
void bayesian_set_game(bayesian_t *game, int i, int j, matrix_t **bimat);
void bayesian_fill_distribution(bayesian_t *game);
void bayesian_fill_distribution_rand(bayesian_t *game);
void bayesian_fill_game(bayesian_t *game, int s);
bayesian_t *generate_random_bayesian(int types, int actions);
void bayesian_print(bayesian_t *game);

#endif
