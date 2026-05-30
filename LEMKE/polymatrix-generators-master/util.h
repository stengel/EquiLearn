#ifndef UTIL_H
#define UTIL_H
/* Generates an integer uniformly at random from [0, n) */
struct matrix_t;
int randint(int n);
double rand_norm (double mu, double sigma);
struct matrix_t *rand_multivariate_single(struct matrix_t *cov, double mu);
struct matrix_t *rand_multivariate_multiple(struct matrix_t *cov, double mu, int reps);
int *init_subset(int *s, int n);
int next_subset(int *s, int *sub, int n);
int myRandom (int size);
void print_int(int *arr, int n);
void print_double(double *v, int n);
int find_draws(int *wins, int k);

#endif
