#include "valuation.h"
#ifndef COMBINATION_H
#define COMBINATION_H

extern int is_multi;
extern int is_unit;
extern int is_budget;
extern int *max_val;
extern int overbidding;
extern int strict;
extern valuation_t *rep_val;
extern int **repetitions;

void print_repetition(int *comb);
int **get_repetitions();
void free_repetitions(int **comb, int n);
int sorted_repetitions(int n, int k);

#endif
