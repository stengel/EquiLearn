#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "valuation.h"
#include "item_bidding.h"
#include "repetition.h"
#include "util.h"

#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x < y) ? x : y)

int tie;

static int win_item(int a1, int a2)
{
    if (a1 > a2)
        return 0;
    else if (a2 > a1)
        return 1;
    return tie;//rand() % 2;
}

static int *win(int *a1, int *a2, int n)
{
    int *wins = malloc(n * sizeof(int));
    int i;

    for (i = 0; i < n; ++i)
        wins[i] = win_item(a1[i], a2[i]);

    return wins;
}

static int get_player_payoff_first(valuation_t *v, int *b1, int *b2,  int k, int *wins, int p)
{
    int j;
    int payoff = 0;

    //payoff = val_func[unit](v, wins, k, p);
    payoff = valuation_get_valuation(v, wins, p);
    //printf("valuation %d\n", payoff);
    for (j = 0; j < k; ++j)
        payoff -= (wins[j] == p) ? (b1[j]) : 0;

    return payoff;
}

static int get_player_payoff_second(valuation_t *v, int *b1, int *b2,  int k, int *wins, int p)
{
    int j;
    int payoff = 0;

    //payoff = val_func[unit](v, wins, k, p);
    payoff = valuation_get_valuation(v, wins, p);
    for (j = 0; j < k; ++j)
        payoff -= (wins[j] == p) ? (b2[j]) : 0;

    return payoff;
}

static int get_player_payoff_allpay(valuation_t *v, int *b1, int *b2, int k, int *wins, int p)
{
    int i;
    int payoff = 0;

    //payoff = val_func[unit](v, wins, k, p);
    payoff = valuation_get_valuation(v, wins, p);
    for (i = 0; i < k; ++i)
        payoff -= b1[i];

    return payoff;
}

static int (*auction[4])(valuation_t *v, int *b1, int *b2,  int k, int *wins, int p) = {
    get_player_payoff_first, get_player_payoff_second, get_player_payoff_allpay
};

static void get_payoff(valuation_t *v1, valuation_t *v2, double *payoffs, int *b1, int *b2, int k, auction_item_t auc_type)
{
    int *wins = win(b1, b2, k);
    int *w_cpy = malloc(sizeof(int) * k);
    memcpy(w_cpy, wins, sizeof(int) * k);

    /*
    printf("P1\n");
    print_int(b1, k);
    printf("P2\n");
    print_int(b2, k);
    printf("C\n");
    print_int(c, 2 * k);
    */

    int d_count = find_draws(wins, k);
    int i, j, l;

    if (d_count > 1) {
        payoffs[0] = 0;
        payoffs[1] = 0;
        max_val = calloc(d_count, sizeof(int));

        for (i = 0; i < d_count; ++i)
            max_val[i] = 1;

        for (i = 0; i < k; ++i)
            if (w_cpy[i] == 2)
                w_cpy[i] = 0;

        int n = sorted_repetitions(2, d_count);
        int **s = get_repetitions();
        int count = 0;
        //printf("Draws %d\n", d_count);
        //print_arr(w_cpy, k);
        //printf("Comb %d\n", n);
        for (i = 0; i < n; ++i) {
            for (j = 0, l = 0; j < k; ++j) {
                if (wins[j] == 2)
                    w_cpy[j] = s[i][l++];
            }
            //print_arr(w_cpy, k);
            count++;
            payoffs[0] += auction[auc_type](v1, b1, b2, k, w_cpy, 0);
            payoffs[1] += auction[auc_type](v2, b2, b1, k, w_cpy, 1);
        }

        payoffs[0] /= count;
        payoffs[1] /= count;
        //printf("Payoff %lf %lf\n", payoffs[0], payoffs[1]);
        free(max_val);
        free_repetitions(s, n);
    }
    else {
        payoffs[0] = auction[auc_type](v1, b1, b2, k, wins, 0);
        payoffs[1] = auction[auc_type](v2, b2, b1, k, wins, 1);
    }
    //printf("b1 ");
    //print_arr(b1, k);
    //printf("b2 ");
    //print_arr(b2, k);
    //printf("%d %d\n", payoffs[0], payoffs[1]);
    free(wins);
}

matrix_t **generate_item_auction(int items, int actions, valuation_t *v1, valuation_t *v2, auction_item_t auction_type)
{
    is_multi = 0;
    /*
    if (unit == 1)
        is_unit = 1;
    if (unit == 3)
        is_budget = 1;
    */
    int m, n, act, **s1, **s2;

    //printf("====\n");
    rep_val = v1;
    max_val = valuation_max_single(v1, &act);
    if (actions)
        act = actions;
    m = sorted_repetitions(act, items);
    s1 = get_repetitions();
    free(max_val);
    //printf("====\n");
    rep_val = v2;
    max_val = valuation_max_single(v2, &act);

    if (actions)
        act = actions;

    n = sorted_repetitions(act, items);
    s2 = get_repetitions();
    free(max_val);
    //printf("====\n");

    //printf("Actions %d items %d total %d\n", actions, items, m);
    //printf("Actions %d items %d total %d\n", actions, items, n);

    matrix_t **game = malloc(2 * sizeof(matrix_t *));
    game[0] = matrix_alloc(m, n);
    game[1] = matrix_alloc(m, n);
    int i, j;
    for (i = 0; i < m; ++i)
        for (j = 0; j < n; ++j) {
            double payoffs[2] = {0, 0};
            get_payoff(v1, v2, payoffs, s1[i], s2[j], items, auction_type);
            game[0]->data[i][j] = payoffs[0];
            game[1]->data[i][j] = payoffs[1];
        }

    free_repetitions(s1, m);
    free_repetitions(s2, n);
    return game;
}

bayesian_t *generate_bayesian_item_auction_val(int types, int items, int actions, char *v, auction_item_t auction_type, int assym)
{
    bayesian_t *game = bayesian_alloc(types, types);
    bayesian_fill_distribution(game);
    is_multi = 0;

    //int **val = generate_players_valuations_rand(types, items, actions);
    //pch = strtok(v, " ");
    //int **val1, **val2;
    valuation_t **val1, **val2;
    //int t = items;
    //int k = 0;
    //if (unit == 2)
    //    t = pow(2, items);
    //if (unit == 3) {
    //    k = t;
    //    t += 1;
    //}
    //val1 = generate_valuation_from_string(v, types, t, 0, k);
    val1 = valuations_read(types, items, v);
    if (assym)
        val2 = valuations_read(types, items, NULL);
        //val2 = generate_valuation_from_string(v, types, t, 0, k);
    else
        val2 = val1;

    int i, j;
    for (i = 0; i < types; ++i) {
        for (j = 0; j < types; ++j) {
            //printf("Valuations\n");
            //print_int(val1[i], items +1);
            //print_int(val2[j], items +1);
            //int m1 = MAX(val1[i][0], val1[j][0]);
            //int m2 = MAX(val2[i][0], val2[j][0]);
            //int k;
            //printf("Items %d", items);
            //for (k = 1; k < items; ++k) {
            //    m1 = MAX(m1,  MAX(val1[i][k], val1[j][k]));
            //    m2 = MAX(m2,  MAX(val2[i][k], val2[j][k]));
                //printf("%d %d %d\n", m, val[i][k], val[j][k]);
            //}
            //printf("M= %d\n", m);
            matrix_t **bimatrix = generate_item_auction(items, actions, val1[i], val2[j], auction_type);
            //matrix_print(bimatrix[0]);
            //printf("==============\n");
            //matrix_print(bimatrix[1]);
            bayesian_set_game(game, i, j, bimatrix);
        }
    }

    return game;
}
