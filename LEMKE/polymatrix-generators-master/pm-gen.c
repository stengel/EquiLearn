#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include "polymatrix.h"
#include "bayesian.h"
#include "multi_unit.h"
#include "item_bidding.h"
#include "blotto.h"
#include "adj_winner.h"
#include "coord_zero.h"
#include "strict_comp.h"
#include "weight_coop.h"

int prec = 15;

polymatrix_t *generate_game(char *game, char *graph, int actions, int g1, int g2, double p)
{
    if(strncmp(game, "CoordZero", 9) == 0) {
        return generate_polymatrix_zero_coord(actions, p, graph, g1, g2);
    } else if(strncmp(game, "GroupZero", 9) == 0) {
        return generate_polymatrix_group_zero(actions, p, graph, g1, g2);
    } else if(strncmp(game, "StrictComp", 10) == 0) {
        return generate_polymatrix_strict_comp(actions, p, graph, g1, g2);
    } else if(strncmp(game, "WeightCoop", 10) == 0) {
        return generate_polymatrix_weight_coop(actions, p, graph, g1, g2);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    char *game, *graph;
    int a, c, g1, g2;
    double p;
    FILE *f;
    srand(time(NULL));
    a = 0;
    g1 = 0;
    g2 = 0;
    p = 0;
    /*
     * a - # of actions
     * b - g1
     * c - g2
     * p - p
     * f - file output
     * v - valuation string
     * r - random seed
     */
    while ((c = getopt(argc, argv, "a:m:n:g:G:f:p:r:")) != -1)
        switch (c)
        {
            case 'a':
                a = atoi(optarg);
                break;
            case 'm':
                g1 = atoi(optarg);
                break;
            case 'n':
                g2 = atoi(optarg);
                break;
            case 'f':
                f = fopen(optarg,"w+");
                break;
            case 'g':
                game = optarg;
                break;
            case 'G':
                graph = optarg;
                break;
            case 'p':
                p = atof(optarg);
                break;
            case 'r':
                srand(atoi(optarg));
                break;
            case '?':
                if (isprint(optopt))
                    fprintf(stderr, "Unknown option '-%c.\n", optopt);
                else
                    fprintf(stderr, "Unknown option character '\\x%x.\n",
                            optopt);
                return 1;
            default:
                break;
        }

    polymatrix_t *pgame = generate_game(game, graph, a, g1, g2, p);
    polymatrix_to_file(f, NULL, pgame);
    printf("%d\n", polymatrix_size(pgame));
    //polymatrix_print(pgame);

    return 0;
}
