#include "matrix.h"
#include "bayesian.h"
#ifndef ADJ_WINNER_H
#define ADJ_WINNER_H

matrix_t **generate_adj_winner(int T1, int T2, double *vals1, double *vals2, int n);
bayesian_t *generate_adj_winner_bayesian_troops(int Tmin, int Tmax, int n, int c, int assym);
bayesian_t *generate_adj_winner_bayesian_payoffs(int types, int T, int n, int c, int assym);
bayesian_t *generate_adj_winner_bayesian_troops_str(int types, int Tmax, int n, int c, char *str, int assym);

#endif
