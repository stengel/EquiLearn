#include <string.h>
#include "util.h"
#include "matrix.h"
#include "adj_winner.h"
#include "partition.h"

#define MIN(x, y) (x < y) ? x : y

static matrix_t *generate_payoff_cov(int n, int c)
{
    matrix_t *cov = matrix_alloc(2, 2);

    cov->data[1][0] = cov->data[0][1] = c;
    cov->data[0][0] = cov->data[1][1] = 1;
    matrix_t *payoff = rand_multivariate_multiple(cov, 0, n);
    matrix_free(cov);

    return payoff;
}

static int win_battle(int a1, int a2)
{
    if (a1 > a2)
        return 0;
    else if (a2 > a1)
        return 1;
    return 0;
}

static double *win(int *a1, int *a2, int n)
{
    double *wins = malloc(n*sizeof(double));
    int i;

    for (i = 0; i < n; ++i)
        wins[i] = win_battle(a1[i], a2[i]);

    return wins;
}

static void get_valuation(double *val1, double *val2, double *payoffs, double *wins, int n)
{
    int i;
    payoffs[0] = 0;
    payoffs[1] = 0;
    for (i = 0; i < n; ++i) {
        payoffs[0] += (1 - wins[i]) * val1[i];
        payoffs[1] += wins[i] * val2[i];
    }
}

typedef struct ratio{
    int idx;
    double ratio;
}ratio_t;

int cmp_ratio(const void *a, const void *b)
{
    ratio_t *r1 = *(ratio_t **) a;
    ratio_t *r2 = *(ratio_t **) b;

    if (r1->ratio != r1->ratio && r2->ratio != r2->ratio)
        return 0;

    if (r1->ratio != r1->ratio)
        return 1;

    if (r2->ratio != r2->ratio)
        return -1;

    if (r1->ratio < r2->ratio)
        return -1;

    if (r1->ratio == r2->ratio)
        return 0;

    return 1;
}

static void get_payoff(double *val1, double *val2, double* payoffs, int *s1, int *s2, int n)
{
    double *wins = win(s1, s2, n);

    double vals[2] = {0, 0};
    get_valuation(val1, val2, vals, wins, n);

    //printf("Vals %lf %lf\n", vals[0], vals[1]);
    int i, c = 0;
    ratio_t **rat = malloc(sizeof(ratio_t *) * n);
    for (i = 0; i < n; ++i) {
        //printf("Wins %lf\n", wins[i]);
        if (wins[i] == 0) {
            rat[c] = malloc(sizeof(ratio_t));
            rat[c]->idx = i;
            rat[c]->ratio = (double)s1[i] / (double) s2[i];
            c++;
            //printf("%d %d %lf\n", c, rat[c-1]->idx, rat[c-1]->ratio);
        }
    }

    qsort(rat, c, sizeof(ratio_t *), cmp_ratio);

    //printf("Sorted\n");
    for (i = 0; i < c; ++i) {
        //printf("%d %d %lf\n", i, rat[i]->idx, rat[i]->ratio);
        double t1 = val1[rat[i]->idx];
        double t2 = val2[rat[i]->idx];
        double v1 = vals[0];
        double v2 = vals[1];
        double p = MIN((v1 - v2) / (t1 + t2), 1);
        //printf("%lf %lf %lf %lf %lf\n", v1, v2, t1, t2, p);
        vals[0] = (v1 - t1) + (t1 * (1 - p));
        vals[1] = v2 + (t2 * p);
        //printf("%lf %lf\n", vals[0], vals[1]);
        free(rat[i]);
    }
    free(rat);

    payoffs[0] = vals[0];
    payoffs[1] = vals[1];
}

matrix_t **generate_adj_winner(int T1, int T2, double *vals1, double *vals2, int n)
{
    int m, k, **s1, **s2;
    m = sorted_partitions(T1, n);
    s1 = get_partitions();

    if (T1 == T2) {
        k = m;
        s2 = s1;
    }
    else {
        k = sorted_partitions(T2, n);
        s2 = get_partitions();
    }

    matrix_t **game = malloc(2*sizeof(matrix_t *));
    game[0] = matrix_alloc(m, k);
    game[1] = matrix_alloc(m, k);

    int i, j;
    for (i = 0; i < m; ++i)
        for (j = 0; j < k; ++j) {
            double payoffs[2] = {0, 0};
            get_payoff(vals1, vals2, payoffs, s1[i], s2[j], n);
            game[0]->data[i][j] = payoffs[0];
            game[1]->data[i][j] = payoffs[1];
        }

    partition_free(s1, m);
    if (T1 != T2)
        partition_free(s2, k);

    return game;
}

/*
matrix_t **generate_adj_winner_pay(int T, int n, double **vals)
{
    int m = sorted_partitions(T, n);
    matrix_t **game = malloc(2*sizeof(matrix_t *));
    game[0] = matrix_alloc(m, m);
    game[1] = matrix_alloc(m, m);

    int i, j;
    for (i = 0; i < m; ++i)
        for (j = 0; j < m; ++j) {
            double payoffs[2] = {0, 0};
            get_payoff(vals[0], vals[1], payoffs, partitions[i], partitions[j], n);
            game[0]->data[i][j] = payoffs[0];
            game[1]->data[i][j] = payoffs[1];
        }

    return game;
}
*/

matrix_t **generate_adj_winner_cov(int T1, int T2, int n, int c)
{
    matrix_t *v = generate_payoff_cov(n, c);
    matrix_t *vals = matrix_trans(v);

    matrix_t **game = generate_adj_winner(T1, T2, vals->data[0], vals->data[1], n);

    matrix_free(vals);
    matrix_free(v);
    return game;
}

static double **generate_players_payoffs_rand(int types, int n)
{
    double **pay = malloc(types * sizeof(double *));
    int i, j;
    for (i = 0; i < types; ++i) {
        pay[i] = malloc(sizeof(double) * n);
        double sum = 0;
        for (j = 0; j < n; ++j) {
            pay[i][j] = (double)rand() / RAND_MAX;
            sum += pay[i][j];
        }
        for (j = 0; j < n; ++j)
            pay[i][j] /= sum;
        //printf("Val\n");
        //for (j = 0; j < n; ++j)
        //    printf("%lf ", pay[i][j]);
        //printf("\n");
    }

    return pay;
}

bayesian_t *generate_adj_winner_bayesian_troops(int Tmin, int Tmax, int n, int c, int assym)
{
    int types = Tmax - Tmin + 1;
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);

    double **val1, **val2;
    //val1  = generate_players_payoffs_rand(types, n);
    val1  = generate_players_payoffs_rand(1, n);
    if (assym) 
        //val2  = generate_players_payoffs_rand(types, n);
        val2  = generate_players_payoffs_rand(1, n);
    else
        val2 = val1;

    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            matrix_t **bimatrix = generate_adj_winner(Tmin + i, Tmin + j, val1[0], val2[0], n);
            //matrix_t *v = generate_payoff_cov(n, c);
            //matrix_t *vals = matrix_trans(v);
            //matrix_t **bimatrix = generate_adj_winner(Tmin + i, Tmin + j, vals->data[i], vals->data[j], n);
            bayesian_set_game(game, i, j, bimatrix);
            //matrix_free(v);
            //matrix_free(vals);
        }
    }

    return game;
}

static char *pch;
static int *generate_troops_from_string(char *str, int types)
{
    int *type = malloc(types * sizeof(int));
    int i;

    //char *s = strtok(str, " ");
    for (i = 0; i < types; ++i) {
        //val[i] = calloc(n, sizeof(int));
        //printf("%d, %d -> %s\n", i, j, pch);
        type[i] = atoi(pch);
        pch = strtok(NULL, " ");
    }
    return type;
}

bayesian_t *generate_adj_winner_bayesian_troops_str(int types, int Tmax, int n, int c, char *str, int assym)
{
    printf("String\n");
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);

    double **val1, **val2;
    pch = strtok(str, " ");
    //val1  = generate_players_payoffs_rand(types, n);
    val1  = generate_players_payoffs_rand(1, n);
    if (assym) 
        //val2  = generate_players_payoffs_rand(types, n);
        val2  = generate_players_payoffs_rand(1, n);
    else
        val2 = val1;

    int *type1, *type2;
    type1 = generate_troops_from_string(str, types);
    if (assym)
        type2 = generate_troops_from_string(str, types);
    else
        type2 = type1;

    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            matrix_t **bimatrix = generate_adj_winner(type1[i], type2[j], val1[0], val2[0], n);
            //matrix_t *v = generate_payoff_cov(n, c);
            //matrix_t *vals = matrix_trans(v);
            //matrix_t **bimatrix = generate_adj_winner(Tmin + i, Tmin + j, vals->data[i], vals->data[j], n);
            bayesian_set_game(game, i, j, bimatrix);
            //matrix_free(v);
            //matrix_free(vals);
        }
    }

    return game;
}

bayesian_t *generate_adj_winner_bayesian_payoffs(int types, int T, int n, int c, int assym)
{
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);

    double **pay1, **pay2;
    pay1 = generate_players_payoffs_rand(types, n);
    if (assym){
        pay2 = generate_players_payoffs_rand(types, n);
    }
    else
        pay2 = pay1;

    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            matrix_t **bimatrix = generate_adj_winner(T, T, pay1[i], pay2[j], n);
            bayesian_set_game(game, i, j, bimatrix);
        }
    }

    return game;
}
