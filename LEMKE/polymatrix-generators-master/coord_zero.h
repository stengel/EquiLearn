#include "matrix.h"
#include "bayesian.h"

#ifndef COORD_ZERO_H
#define COORD_ZERO_H

#define BAYESIAN_ZERO 0
#define BAYESIAN_COORD 1

matrix_t **generate_zero(int m, int n);
matrix_t **generate_coord(int m, int n);
bayesian_t *generate_bayesian_zero(int types, int actions);
bayesian_t *generate_bayesian_coord(int types, int actions);
bayesian_t *generate_bayesian_zero_coord(int types, int actions, double p);
polymatrix_t *generate_polymatrix_zero_coord(int actions, double p, char *graph, int g1, int g2);
polymatrix_t *generate_polymatrix_group_zero(int actions, double p, char *graph, int g1, int g2);

#endif
