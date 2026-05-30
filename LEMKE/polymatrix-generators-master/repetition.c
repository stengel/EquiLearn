#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "repetition.h"

int *max_val;
int **repetitions = NULL;
int is_multi = 0;
int is_unit;
int is_budget;
int overbidding = 0;
static int max;
static int gn;
static int count = 0;
int strict = 0;
valuation_t *rep_val;

int is_overbidding(int *comb)
{
    int i;
    for (i = 0; i < gn; i++){
        if (comb[i] > max_val[i])
            return 1;
        else if (strict && comb[i] == max_val[i] && max_val[i] != 0)
            return 1;
    }
    return 0;
}

int is_overbidding_unit(int *comb)
{
    int i, j, max, sum = 0;
    j = -1;
    max = 0;
    for (i = 0; i < gn; i++)
        if (comb[i] > 0 && max_val[i] > max){
            max = max_val[i];
            j = i;
        }

    for (i = 0; i < gn; i++)
        sum += comb[i];

    return sum > max;
}

int is_overbidding_budget(int *comb)
{
    int i, sum = 0;

    for (i = 0; i < gn; i++)
        sum += comb[i];

    return sum > max_val[gn];
}

int is_invalid_multi(int *comb)
{
    int i;
    for (i = 0; i < gn - 1; i++)
        if (comb[i] < comb[i+1]) {
            return 1;
        }
    return 0;
}

void add_repetition(int *comb)
{
    if (!overbidding && valuation_is_overbidding(rep_val, max_val, comb))
        return;
    /*
    if (is_overbidding(comb))
        return;

    if (is_unit && is_overbidding_unit(comb))
        return;

    if (is_budget && is_overbidding_budget(comb))
        return;
    */

    if (is_multi && is_invalid_multi(comb)) 
        return;

    count++;
    int **tmp = realloc(repetitions, count * sizeof(int *));
    if (!tmp)
        exit(1);
    repetitions = tmp;

    repetitions[count - 1] = calloc(gn, sizeof(int));
    memcpy(repetitions[count - 1], comb, gn * sizeof(int));
}

int **get_repetitions()
{
    int **copy = repetitions;
    repetitions = NULL;
    return copy;
}

void free_repetitions(int **comb, int n)
{
    int i;
    for (i = 0; i < n; i++)
        free(comb[i]);
    free(comb);
}

int should_stop(int *c, int m, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (overbidding && c[i] != m)
            return 0;
        else if (!overbidding && c[i] != max_val[i])
            return 0;
    }


    return 1;
}

int next_rep(int *c, int m, int n)
{
    if (should_stop(c, m, n))
        return 0;

    int i, carry = 1;
    int d = m + 1;
    for (i = 0; i < n; ++i) {
        int val = c[i] + carry;
        if (!overbidding)
            d = max_val[i] + 1;
        carry = val / d;
        val %= d;
        c[i] = val;
        if (carry == 0)
            break;
    }
    return 1;
}

void reps(int m, int n, int *c)
{
    do {
        add_repetition(c);
        //printf("Adding ");
        //print_repetition(c);
    } while (next_rep(c, m, n));
}

void print_repetition(int *comb)
{
    int j;
    for (j = 0; j < gn; j++)
        printf("%d ", comb[j]);
    printf("\n");
}

static int max_arr(int *arr, int n)
{
    int i;
    int max_v = arr[0];
    for (i = 1; i < n; ++i)
        max_v = (max_v > arr[i]) ? max_v : arr[i];
    return max_v;
}
int sorted_repetitions(int m, int n)
{
    if (repetitions)
        free_repetitions(repetitions, count);

    if (!max_val) {
        max_val = malloc(sizeof(int) * n);
        int i;
        for (i = 0; i < n; ++i)
            max_val[i] = INT_MAX;
    }
    max = max_arr(max_val, n);

    repetitions = NULL;
    count = 0;
    gn = n;
    int *c = calloc(n, sizeof(int));
    reps(m, n, c);
    free(c);

    return count;
}
