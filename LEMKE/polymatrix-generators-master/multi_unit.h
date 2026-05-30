#include "bayesian.h"

#ifndef MULTI_UNIT_H
#define MULTI_UNIT_H

extern int tie;

typedef enum auction
{
    UNIFORM,
    DUTCH,
    FIRST_PRICE,
    ALL_PAY,
    SECOND_PRICE,
}auction_t;

//matrix_t **generate_auction(int items, int actions, int *v1, int *v2, auction_t auction_type);
bayesian_t *generate_bayesian_auction(int types, int items, int actions, auction_t auction_type, int strict, int assym);
bayesian_t *generate_bayesian_auction_val(int types, int items, char *v, auction_t auction_type, int strict, int assym);

#endif
