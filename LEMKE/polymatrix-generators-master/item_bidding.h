#include "bayesian.h"

#ifndef ITEM_BIDDING_H 
#define ITEM_BIDDING_H

extern int tie;

typedef enum auction_item
{
    FIRST_PRICE_ITEM,
    SECOND_PRICE_ITEM,
    ALL_PAY_ITEM,
}auction_item_t;

//matrix_t **generate_auction(int items, int actions, double *v1, double *v2, auction_item_t auction_type);
bayesian_t *generate_bayesian_item_auction(int types, int items, int actions, auction_item_t auction_type, int assym, int unit);
bayesian_t *generate_bayesian_item_auction_val(int types, int items, int actions, char *v, auction_item_t auction_type, int assym);

#endif
