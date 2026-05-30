#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "multi_unit.h"
#include "repetition.h"

#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x < y) ? x : y)

int tie;

static int win_item(int a1, int a2)
{
    if (a1 > a2)
        return 0;
    else if (a2 > a1)
        return 1;
    return tie;
}

static int *win(int *a1, int *a2, int n)
{
    int *wins = malloc(n * sizeof(int));
    int i, j, k;

    j = 0; k = 0;
    for (i = 0; i < n; ++i) {
        wins[i] = win_item(a1[j], a2[k]);

        if (wins[i] == 0)
            j++;
        else
            k++;
    }

    return wins;
}

int **win_recursive(int *a1, int *a2, int *w, int **wins, int *r, int n, int s, int j, int k)
{
    int i;
    for (i = s; i < n; ++i) {
        w[i] = win_item(a1[j], a2[k]);

        if (w[i] == 0)
            j++;
        else if (w[i] == 1)
            k++;
        else {
            w[i] = 0;
            wins = win_recursive(a1, a2, w, wins, r, n, i + 1, j + 1, k);
            w[i] = 1;
            wins = win_recursive(a1, a2, w, wins, r, n, i + 1, j, k + 1);
            break;
        }
    }

    if (i == n) {
        *r += 1;
        int **n_wins = realloc(wins, *r * sizeof(int *));
        if (n_wins != NULL) {
            wins = n_wins;
            wins[*r - 1] = malloc(sizeof(int) * n);
            memcpy(wins[*r-1], w, sizeof(int) * n);
        } else {
            free(wins);
            puts("Error (re)allocating memory\n");
            exit(1);
        }
    }
    return wins;
}

int total_won_by_p2(int *wins, int n)
{
    int i, total;
    
    total = 0;
    for (i = 0; i < n; i++)
        total += wins[i];

    return total;
}

int *sort_bids(int *b1, int *b2, int n)
{
    int i, j, k;
    int *c = malloc(sizeof(int) * (2 * n));

    j = k = 0;
    for (i = 0; i < 2 * n; i++) {
        if (b1[j] > b2[k])
            c[i] = b1[j++];
        else
            c[i] = b2[k++];
    }

    return c;
}

int get_player_payoff_uniform(int *v, int *b1, int *b2, int *c, int J, int k)
{
    int j;
    int payoff = 0;

    int price = 0;

    if (J == 0)
        price = b1[0];
    else if (J == k)
        price = b2[0];
    else
        price = MAX(b1[J], b2[k - J]);

    for (j = 0; j < J; ++j)
        payoff += (v[j] - price);

    return payoff;
}

int get_player_payoff_dutch(int *v, int *b1, int *b2, int *c, int J, int k)
{
    int j;
    int payoff = 0;

    for (j = 0; j < J; ++j)
        payoff += (v[j] - fmax(b1[j], c[k -J]));

    return payoff;
}

int get_player_payoff_first(int *v, int *b1, int *b2, int *c, int J, int k)
{
    int j;
    int payoff = 0;

    for (j = 0; j < J; ++j)
        payoff += (v[j] - b1[j]);

    return payoff;
}

int get_player_payoff_allpay(int *v, int *b1, int *b2, int *c, int J, int k)
{
    int i;
    int payoff = 0;

    for (i = 0; i < J; ++i)
        payoff += (v[i] - b1[i]);

    for (; i < k; ++i)
        payoff -= b1[i];

    return payoff;
}

static int (*auction[4])(int *v, int *b1, int *b2, int *c, int J, int k) = {
    get_player_payoff_first,get_player_payoff_uniform,
    get_player_payoff_allpay, get_player_payoff_dutch
};

static void get_payoff(int *v1, int *v2, double *payoffs, int *b1, int *b2, int k, auction_t auc_type)
{
    int *wins = win(b1, b2, k);
    int J2 = total_won_by_p2(wins, k);
    int J1 = k - J2;
    int *c = sort_bids(b1, b2, k);

    payoffs[0] = auction[auc_type](v1, b1, b2, c, J1, k);
    payoffs[1] = auction[auc_type](v2, b2, b1, c, J2, k);

    free(c);
}

static void get_payoff_rec(int *v1, int *v2, double *payoffs, int *b1, int *b2, int k, auction_t auc_type)
{
    int **wins = NULL;
    int *w = malloc(k * sizeof(int));
    int d_count = 0;
    wins = win_recursive(b1, b2, w, wins, &d_count, k, 0, 0, 0);
    int i;
    payoffs[0] = 0;
    payoffs[1] = 0;
    for (i = 0; i < d_count; i++) {
        int J2 = total_won_by_p2(wins[i], k);
        int J1 = k - J2;
        int *c = sort_bids(b1, b2, k);

        payoffs[0] += auction[auc_type](v1, b1, b2, c, J1, k);
        payoffs[1] += auction[auc_type](v2, b2, b1, c, J2, k);

        free(c);
    }

    payoffs[0] /= d_count;
    payoffs[1] /= d_count;
}

matrix_t **generate_auction(int items, int actions, int *v1, int *v2, auction_t auction_type, int strict)
{
    int m, n, **s1, **s2;
    is_multi = 1;
    max_val = v1;
    m = sorted_repetitions(actions, items);
    s1 = get_repetitions();
    max_val = v2;
    n = sorted_repetitions(actions, items);
    s2 = get_repetitions();

    matrix_t **game = malloc(2 * sizeof(matrix_t *));
    game[0] = matrix_alloc(m, n);
    game[1] = matrix_alloc(m, n);
    int i, j;
    for (i = 0; i < m; ++i)
        for (j = 0; j < n; ++j) {
            double payoffs[2] = {0, 0};
            if (tie == 2)
                get_payoff_rec(v1, v2, payoffs, s1[i], s2[j], items, auction_type);
            else
                get_payoff(v1, v2, payoffs, s1[i], s2[j], items, auction_type);
            game[0]->data[i][j] = payoffs[0];
            game[1]->data[i][j] = payoffs[1];
        }

    free_repetitions(s1, m);
    return game;
}

int **generate_players_valuations_rand(int types, int n, int actions)
{
    int **val = malloc(types * sizeof(int *));
    int i, j;
    for (i = 0; i < types; ++i) {
        val[i] = malloc(sizeof(int) * n);
        for (j = 0; j < n; ++j)
            //val[i][j] = actions;
            //val[i][j] = ((double)rand() / RAND_MAX) * actions;
            val[i][j] = rand() % actions;
    }

    return val;
}

static char *pch;
static int **generate_valuation_from_string(char *str, int types, int n)
{
    int **val = malloc(types * sizeof(int *));
    int i, j;

    //char *s = strtok(str, " ");
    for (i = 0; i < types; ++i) {
        val[i] = calloc(n, sizeof(int));
        for (j = 0; j < n; ++j) {
            val[i][j] = atoi(pch);
            pch = strtok(NULL, " ");
        }
    }
    return val;
}

bayesian_t *generate_bayesian_auction(int types, int items, int actions, auction_t auction_type, int strict, int assym)
{
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);

    int **val1, **val2;
    val1 = generate_players_valuations_rand(types, items, actions);
    if (assym)
        val2 = generate_players_valuations_rand(types, items, actions);
    else
        val2 = val1;
    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            matrix_t **bimatrix = generate_auction(items, actions, val1[i], val2[j], auction_type, strict);
            
            //printf("Bimatrix type %d %d\n", i, j);
            //matrix_print(bimatrix[0]);
            //printf("---\n");
            //matrix_print(bimatrix[1]);
            //printf("===============\n");

            bayesian_set_game(game, i, j, bimatrix);
        }
    }

    return game;
}

bayesian_t *generate_bayesian_auction_val(int types, int items, char *v, auction_t auction_type, int strict, int assym)
{
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);

    //int **val = generate_players_valuations_rand(types, items, actions);
    pch = strtok(v, " ");
    int **val1, **val2;
    val1 = generate_valuation_from_string(v, types, items);
    if (assym)
        val2 = generate_valuation_from_string(v, types, items);
    else
        val2 = val1;
    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            //printf("Valuations\n");
            //print_int(val[i], items);
            //print_int(val[j], items);
            int m = MAX(val1[i][0], val2[j][0]);
            int k;
            for (k = 1; k < items; ++k) {
                m = MAX(m,  MAX(val1[i][k], val2[j][k]));
                //printf("%d %d %d\n", m, val[i][k], val[j][k]);
            }
            matrix_t **bimatrix = generate_auction(items, m + 1, val1[i], val2[j], auction_type, strict);
            
            //printf("Bimatrix type %d %d\n", i, j);
            //matrix_print(bimatrix[0]);
            //printf("---\n");
            //matrix_print(bimatrix[1]);
            //printf("===============\n");

            bayesian_set_game(game, i, j, bimatrix);
        }
    }

    return game;
}
