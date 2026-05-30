#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include "polymatrix.h"
#include "bayesian.h"
#include "blotto.h"
#include "adj_winner.h"

int prec = 15;

bayesian_t *generate_game(char *game, int types, int troops, int hills, int d, double cov, char *val, int assym)
{
    if (strncmp(game, "Random", 6) == 0) {
        return generate_random_bayesian(types, troops);
    }
    else if (strncmp(game, "BlottoTroops", 12) == 0) {
        if (d > 0)
            return generate_blotto_bayesian_troops(types, troops, hills, d, assym);
        return generate_blotto_bayesian_troops_str(types, troops, hills, d, val, assym);
    }
    else if (strncmp(game, "BlottoPayoffCov", 15) == 0) {
        return generate_blotto_bayesian_payoffs_cov(types, troops, hills, cov, assym);
    }
    else if (strncmp(game, "BlottoPayoff", 12) == 0) {
        return generate_blotto_bayesian_payoffs(types, troops, hills, d, assym);
    }
    else if (strncmp(game, "AdjWinner", 9) == 0) {
        return generate_adj_winner_bayesian_payoffs(types, troops, hills, d, 1);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    char *game;
    int types, troops, c, hills, d, assym, z;
    double C;
    FILE *f;
    srand(time(NULL));
    char *val;
    hills = -1;
    assym = 1;
    z = 0;
    C = 0;
    /*
     * a - # of types
     * b - # of items
     * c - # of actions / something else
     * C - Covariant
     * f - file output
     * n - assymmetric
     * v - valuation string
     * r - random seed
     * z - Random distribution
     */
    //while ((c = getopt(argc, argv, "a:b:c:C:g:sf:v:mnr:z")) != -1)
    while ((c = getopt(argc, argv, "a:c:f:g:i:r:St:v:z")) != -1)
        switch (c)
        {
            case 'a':
                troops = atoi(optarg);
                break;
            case 't':
                types = atoi(optarg);
                break;
            case 'f':
                f = fopen(optarg,"w+");
                break;
            case 'i':
                hills = atoi(optarg);
                break;
            case 'c':
                C = atof(optarg);
                break;
            case 'S':
                assym = 0;
                break;
            case 'g':
                game = optarg;
                break;
            case 'r':
                srand(atoi(optarg));
                break;
            case 'v':
                d = 1;
                val = optarg;
                break;
            case 'z':
                z = 1;
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

    dist_type = z;
    bayesian_t *bgame = generate_game(game, types, troops, hills, d, C, val, assym);
    polymatrix_t *pgame = bayesian_to_polymatrix(bgame);
    polymatrix_to_file(f, NULL, pgame);
    printf("%d\n", polymatrix_size(pgame));
    bayesian_free(bgame);
    polymatrix_free(pgame);

    return 0;
}
