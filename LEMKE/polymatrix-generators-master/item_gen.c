#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include "valuation.h"
#include "item_bidding.h"
#include "multi_unit.h"
#include "polymatrix.h"
#include "bayesian.h"
#include "util.h"
#include "repetition.h"

int main(int argc, char **argv)
{
    int auction_type, actions, assym, distribution, items, types;
    char c, *val_str, *game;
    FILE *f;

    actions = distribution = items = types = 0;
    assym = 1;
    tie = 0;

    /*
     * A - Auction type
     * a - # of actions
     * d - distribution
     * f - File name
     * i - # of items
     * o - overbidding
     * r - random seed
     * s - stricter overbidding
     * S - symmetric bidders
     * t - # of types
     * T - Tie breaking rule
     * v - string containing all valuation functions
     */
    //while ((c = getopt(argc, argv, "A:a:d:f:g:i:or:sSt:T:v:")) != -1)
    while ((c = getopt(argc, argv, "A:a:d:f:g:i:r:St:T:v:")) != -1)
        switch (c)
        {
            case 'A':
                auction_type = atoi(optarg);
                break;
            case 'a':
                actions = atoi(optarg);
                break;
            case 'd':
                distribution = atoi(optarg);
                break;
            case 'f':
                f = fopen(optarg, "w");
                break;
            case 'g':
                game = optarg;
                break;
            case 'i':
                items = atoi(optarg);
                break;
            case 'o':
                overbidding = 1;
                break;
            case 'r':
                srand(atoi(optarg));
                break;
            case 's':
                strict = 1;
                break;
            case 'S':
                assym = 0;
                break;
            case 't':
                types = atoi(optarg);
                break;
            case 'T':
                tie = atoi(optarg);
                break;
            case 'v':
                val_str = optarg;
                break;
        }

    actions = 0;
    bayesian_t *b_game;

    if (strncmp(game, "Itembidding", 11) == 0)
        b_game = generate_bayesian_item_auction_val(types, items, actions, val_str, auction_type, assym);
    else if (strncmp(game, "Multiunit", 9) == 0)
        b_game = generate_bayesian_auction_val(types, items, val_str, auction_type, 0, assym);
    else
        exit(1);

    polymatrix_t *poly = bayesian_to_polymatrix(b_game);
    polymatrix_to_file(f, NULL, poly);
    printf("%d\n", polymatrix_size(poly));
    bayesian_free(b_game);
    polymatrix_free(poly);

    return 0;
}
