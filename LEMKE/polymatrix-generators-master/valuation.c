#include <math.h>
#include "valuation.h"
#include "repetition.h"
#include "util.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

char *v_pch;
void valuation_fill_str(valuation_t *val)
{
    int i;
    val->val = calloc(val->n, sizeof(int));
    for (i = 0; i < val->n; ++i) {
        v_pch =  strtok(NULL, " ");
        val->val[i] = atoi(v_pch);
    }
}

void valuation_fill(valuation_t *val, char *v_str)
{
    switch(val->type) {
        case UNIT_DEMAND:
        case ADDITIVE:
            val->n = val->items;
            break;
        case SINGLE_MINDED:
        case BUDGET_ADDITIVE:
            val->n = val->items + 1;
            break;
        case SUBMODULAR:
            val->n = pow(2, val->items);
            break;
        default:
            break;
    }
    valuation_fill_str(val);
}

valuation_t *valuation_alloc(int items, val_func_t type, char *v_str)
{
    valuation_t *val = malloc(sizeof(valuation_t));

    val->items = items;
    val->type = type;
    valuation_fill(val, v_str);

    return val;
}

valuation_t **valuations_read(int types, int items, char *v_str)
{
    int i;
    valuation_t **vals = malloc(types * sizeof(valuation_t));
    v_pch = strtok(v_str, " ");
    vals[0] = valuation_alloc(items, atoi(v_pch), v_str);
    //valuation_print(vals[0]);

    for (i = 1; i < types; ++i) {
        v_pch = strtok(NULL, " ");
        //printf("Reading %d %s\n", i, v_pch);
        vals[i] = valuation_alloc(items, atoi(v_pch), v_str);
        //valuation_print(vals[i]);
    }

    return vals;
}

int valuation_submodular(valuation_t *val, int *win)
{
    int i, idx = 0;
    for (i = val->items - 1; i >= 0; i--) {
        idx <<= 1;
        idx += win[i] - 0;
    }
    return val->val[idx];
}

int valuation_unit_demand(valuation_t *val, int *win)
{
    int valuation = 0;
    int i;

    for (i = 0; i < val->items; i++)
        if (win[i])
            valuation = (val->val[i] > valuation) ? val->val[i] : valuation;

    return valuation;
}

int valuation_single_minded(valuation_t *val, int *win)
{
    int i;

    for (i = 0; i < val->items; i++) {
        if (val->val[i] > 0 && !win[i])
            return 0;
    }

    return val->val[val->n - 1];
}

int valuation_additive(valuation_t *val, int *win)
{
    int valuation = 0;
    int i;

    for (i = 0; i < val->items; i++)
        if (win[i])
            valuation += val->val[i];
        
    return valuation;
}

int valuation_budget_additive(valuation_t *val, int *win)
{
    int valuation;

    valuation = valuation_additive(val, win);
    valuation = (valuation > val->val[val->n - 1]) ? val->val[val->n - 1] : valuation;

    return valuation;
}

static int (*valuation_get_val[5])(valuation_t *val, int *win) = {valuation_unit_demand, valuation_single_minded, valuation_additive, valuation_budget_additive, valuation_submodular};

int valuation_get_valuation(valuation_t *val, int *win, int player)
{
    if (!player) {
        int *w = malloc(val->items * sizeof(int));
        int i;
        for (i = 0; i < val->items; ++i)
            w[i] = !win[i];
        int v = valuation_get_val[val->type](val, w);
        free(w);
        return v;
    }
    return valuation_get_val[val->type](val, win);
}
void valuation_free(valuation_t *val)
{
    free(val->val);
    free(val);
}

int *valuation_max_single(valuation_t *val, int *m)
{
    int i;
    int *m_val = calloc(val->items, sizeof(int));
    if (m != NULL)
        *m = 0;

    for (i = 0; i < val->items; i++) {
        switch (val->type) {
            case UNIT_DEMAND:
            case ADDITIVE:
                m_val[i] = val->val[i];
                break;
            case BUDGET_ADDITIVE:
                m_val[i] = (val->val[i] > val->val[val->n - 1]) ? val->val[val->n - 1] : val->val[i];
                break;
            case SINGLE_MINDED:
                m_val[i] = (val->val[i]) ? val->val[val->n - 1] : 0;
                break;
            case SUBMODULAR:
                m_val[i] = val->val[(int)(pow(2, i)) - 1];
                break;
        }
        if (m != NULL)
            *m = (m_val[i] > *m) ? m_val[i] : *m;
    }

    return m_val;
}

void print_arr(int *arr, int k)
{
    int i;
    for (i = 0; i < k; ++i)
        printf("%d ", arr[i]);
    printf("\n");
}

void valuation_print(valuation_t *val)
{
    printf("Valuation Func: %d\n", val->type);
    print_arr(val->val, val->n);

    int *m_val = valuation_max_single(val, NULL);
    print_arr(m_val, val->items);
    int i, j;
    int m = (int)(pow(2, val->items));
    int *s1 = calloc(val->items, sizeof(int));

    for (i = 0; i < m; ++i) {
        for (j = 0; j < val->items; j++)
            s1[j] = (i >> j) & 1;
        printf("%d --> ", valuation_get_valuation(val, s1, 0));
        printf("%d --> ", valuation_get_valuation(val, s1, 1));
        print_arr(s1, val->items);
    }
}

int valuation_is_overbidding_single(valuation_t *val, int *max_s, int *bid)
{
    int i, s;

    s = 0;
    if (strict)
        s = 1;

    for (i = 0; i < val->items; ++i) {
        if (max_s[i] == 0 && bid[i] == 0)
            continue;
        if (bid[i] + s > max_s[i])
            return 1;
    }

    return 0;
}

int valuation_is_overbidding(valuation_t *val, int *max_s, int *bid)
{
    if (val->type == SINGLE_MINDED)
        return valuation_is_overbidding_single(val, max_s, bid);

    int i, sum_bid, s;
    int *sub = init_subset(bid, val->items);

    s = 0;
    if (strict)
        s = 1;

    do {
        int v = valuation_get_valuation(val, sub, 1);
        sum_bid = 0;

        for (i = 0; i < val->items; ++i)
            sum_bid += sub[i];

        if (sum_bid == 0)
            continue;

        if (v < sum_bid + s) {
            //valuation_print(val);
            //printf("Overbidding bid");
            //print_arr(bid, val->items);
            //printf("Subset ");
            //print_arr(sub, val->items);
            //printf("Valuation %d\nSum bid %d\n", v, sum_bid);
            return 1;
        }
    } while (next_subset(bid, sub, val->items));

    return 0;
}
