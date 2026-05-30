#include <string.h>
#include "util.h"
#include "matrix.h"
#include "blotto.h"
#include "partition.h"

matrix_t *generate_payoff_cov(int n, double c)
{
    matrix_t *cov = matrix_alloc(2, 2);

    cov->data[1][0] = cov->data[0][1] = c;
    cov->data[0][0] = cov->data[1][1] = 1;
    matrix_t *payoff = rand_multivariate_multiple(cov, 0.5, n);
    matrix_free(cov);

    return payoff;
}

static int win_battle(int a1, int a2)
{
    if (a1 > a2)
        return 0;
    else if (a2 > a1)
        return 1;
    return 0;//rand() % 2;
}

static int *win(int *a1, int *a2, int n)
{
    int *wins = malloc(n*sizeof(int));
    int i;

    for (i = 0; i < n; ++i)
        wins[i] = win_battle(a1[i], a2[i]);

    return wins;
}

static void get_payoff(double *val1, double *val2, double* payoffs, int *s1, int *s2, int n)
{
    int *wins = win(s1, s2, n);

    int i;
    for (i = 0; i < n; ++i){
        payoffs[wins[i]] += (wins[i] == 0) ? val1[i] : val2[i];
    }

    free(wins);
}

matrix_t **generate_blotto(int T1, int T2, double *vals1, double *vals2, int n)
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

    //vals1[0] = 0.836;
    //vals1[1] = 0.944;
    //vals1[2] = -0.39;
    //vals2[0] = 0.203;  
    //vals2[1] = 0.251; 
    //vals2[2] =-1.405;

    //printf("%.3lf %.3lf %.3lf\n", vals1[0], vals1[1], vals1[2]);
    //printf("%.3lf %.3lf %.3lf\n", vals2[0], vals2[1], vals2[2]);
    matrix_t **game = malloc(2*sizeof(matrix_t *));
    game[0] = matrix_alloc(m, k);
    game[1] = matrix_alloc(m, k);

    int i, j;
    for (i = 0; i < m; ++i)
        for (j = 0; j < k; ++j) {
            double payoffs[2] = {0, 0};
            //print_int(s1[i], 3);
            //print_int(s2[j], 3);
            get_payoff(vals1, vals2, payoffs, s1[i], s2[j], n);
            game[0]->data[i][j] = payoffs[0];
            game[1]->data[i][j] = payoffs[1];
            //printf("%.3lf %.3lf\n", payoffs[0], payoffs[1]);
        }

    partition_free(s1, m);
    if (T1 != T2)
        partition_free(s2, k);

    return game;
}

/*
matrix_t **generate_blotto_pay(int T, int n, double **vals)
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

matrix_t **generate_blotto_cov(int T1, int T2, int n, double c)
{
    matrix_t *v = generate_payoff_cov(n, c);
    matrix_t *vals = matrix_trans(v);

    matrix_t **game = generate_blotto(T1, T2, vals->data[0], vals->data[1], n);

    matrix_free(vals);
    matrix_free(v);
    return game;
}

double ***generate_players_payoffs_cov(int types, int n, double c)
{
    double ***pay = malloc(2*sizeof(int**));
    pay[0] = malloc(types * sizeof(double *));
    pay[1] = malloc(types * sizeof(double *));

    int i;
    for (i = 0; i < types; ++i) {
        pay[0][i] = malloc(sizeof(double) * n);
        pay[1][i] = malloc(sizeof(double) * n);

        matrix_t *v = generate_payoff_cov(n, c);
        matrix_t *vals = matrix_trans(v);
        matrix_print(vals);

        memcpy(pay[0][i], vals->data[0], sizeof(double) * n);
        memcpy(pay[1][i], vals->data[1], sizeof(double) * n);

        matrix_free(vals);
        matrix_free(v);
    }

    return pay;
}

double **generate_players_payoffs_rand(int types, int n)
{
    double **pay = malloc(types * sizeof(double *));
    int i, j;
    for (i = 0; i < types; ++i) {
        pay[i] = malloc(sizeof(double) * n);
        for (j = 0; j < n; ++j)
            pay[i][j] = (double)rand() / RAND_MAX;
    }

    return pay;
}

bayesian_t *generate_blotto_bayesian_troops(int Tmin, int Tmax, int n, int c, int assym)
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
            matrix_t **bimatrix = generate_blotto(Tmin + i, Tmin + j, val1[0], val2[0], n);
            //matrix_t *v = generate_payoff_cov(n, c);
            //matrix_t *vals = matrix_trans(v);
            //matrix_t **bimatrix = generate_blotto(Tmin + i, Tmin + j, vals->data[i], vals->data[j], n);
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

bayesian_t *generate_blotto_bayesian_troops_str(int types, int Tmax, int n, int c, char *str, int assym)
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
            matrix_t **bimatrix = generate_blotto(type1[i], type2[j], val1[0], val2[0], n);
            //matrix_t *v = generate_payoff_cov(n, c);
            //matrix_t *vals = matrix_trans(v);
            //matrix_t **bimatrix = generate_blotto(Tmin + i, Tmin + j, vals->data[i], vals->data[j], n);
            bayesian_set_game(game, i, j, bimatrix);
            //matrix_free(v);
            //matrix_free(vals);
        }
    }

    return game;
}

bayesian_t *generate_blotto_bayesian_payoffs(int types, int T, int n, int c, int assym)
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
            matrix_t **bimatrix = generate_blotto(T, T, pay1[i], pay2[j], n);
            bayesian_set_game(game, i, j, bimatrix);
        }
    }

    for (i = 0; i < types; ++i) {
        free(pay1[i]);
        if (assym)
            free(pay2[i]);
    }
    free(pay1);
    if (assym)
        free(pay2);

    return game;
}

bayesian_t *generate_blotto_bayesian_payoffs_cov(int types, int T, int n, double c, int assym)
{
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);

    double ***pay;
    pay = generate_players_payoffs_cov(types, n, c);

    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            matrix_t **bimatrix = generate_blotto(T, T, pay[0][i], pay[1][j], n);
            bayesian_set_game(game, i, j, bimatrix);
        }
    }

    for (i = 0; i < types; ++i) {
        free(pay[0][i]);
        free(pay[1][i]);
    }

    free(pay[0]);
    free(pay[1]);
    free(pay);

    return game;
}
