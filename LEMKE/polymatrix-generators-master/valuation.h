#ifndef VALUATION_H
#define VALUATION_H

typedef enum {
    UNIT_DEMAND,
    SINGLE_MINDED,
    ADDITIVE,
    BUDGET_ADDITIVE,
    SUBMODULAR
}val_func_t;

typedef struct
{
    val_func_t type;
    int n;
    int items;
    int *val;
}valuation_t;

extern char *v_pch;
valuation_t *valuation_alloc(int items, val_func_t type, char *v_str);
valuation_t **valuations_read(int types, int items, char *v_str);
int *valuation_max_single(valuation_t *val, int *m);
int valuation_get_valuation(valuation_t *val, int *win, int player);
void valuation_free(valuation_t *val);
void valuation_print(valuation_t *val);
void print_arr(int *arr, int k);
int valuation_is_overbidding(valuation_t *val, int *max_s, int *bid);

#endif
