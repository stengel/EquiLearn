#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include "polymatrix.h"
#include "bayesian.h"
#include "ranking.h"
#include "multi_unit.h"
#include "item_bidding.h"
#include "blotto.h"
#include "adj_winner.h"

int prec = 15;

bayesian_t *generate_game(char *game, int a, int b, int c, int d, double cov, char *val, int strict, int assym, int unit)
{
    if(strncmp(game, "MultiUnit", 9) == 0) {
        if (c > 0)
            return generate_bayesian_auction(a, b, c, d, strict, assym);
        return generate_bayesian_auction_val(a, b, val, d, strict, assym);
    }
    else if (strncmp(game, "Item", 4) == 0) {
        if (c > 0)
            return generate_bayesian_item_auction(a, b, c, d, strict, assym, unit);
        return generate_bayesian_item_auction_val(a, b, val, d, strict, assym, unit);
    }
    else if (strncmp(game, "Ranking", 7) == 0) {
        return NULL;//generate_ranking(s);
    }
    else if (strncmp(game, "Random", 6) == 0) {
        return generate_random_bayesian(a, b);
    }
    else if (strncmp(game, "BlottoTroops", 12) == 0) {
        if (d > 0)
            return generate_blotto_bayesian_troops(a, b, c, d, assym);
        return generate_blotto_bayesian_troops_str(a, b, c, d, val, assym);
    }
    else if (strncmp(game, "BlottoPayoffCov", 15) == 0) {
        return generate_blotto_bayesian_payoffs_cov(a, b, c, cov, assym);
    }
    else if (strncmp(game, "BlottoPayoff", 12) == 0) {
        return generate_blotto_bayesian_payoffs(a, b, c, d, assym);
    }
    else if (strncmp(game, "AdjWinner", 9) == 0) {
        return generate_adj_winner_bayesian_payoffs(a, b, c, d, 1);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    char *game;
    int a, b, B, c,c1, d, s, n, u, z, m;
    double C;
    FILE *f;
    srand(time(NULL));
    char *val;
    c1 = -1;
    n = 0;
    s = 0;
    u = 0;
    z = 0;
    m = 0;
    B = 0;
    C = 0;
    /*
     * a - # of types
     * b - # of items
     * c - # of actions / something else
     * C - Covariant
     * f - file output
     * m - submodular
     * n - assymmetric
     * v - valuation string
     * r - random seed
     * s - strict overbidding
     * u - Unit valuation
     * z - Random distribution
     */
    while ((c = getopt(argc, argv, "a:b:c:C:d:g:sf:v:mnr:uzB")) != -1)
        switch (c)
        {
            case 'a':
                a = atoi(optarg);
                break;
            case 'B':
                B = 1;
                break;
            case 'f':
                f = fopen(optarg,"w+");
                break;
            case 'b':
                b = atoi(optarg);
                break;
            case 'c':
                c1 = atoi(optarg);
                break;
            case 'C':
                C = atof(optarg);
                break;
            case 'n':
                n = 1;
                break;
            case 'g':
                game = optarg;
                break;
            case 'm':
                m = 1;
                break;
            case 'r':
                srand(atoi(optarg));
                break;
            case 's':
                s = 1;
                break;
            case 'd':
                d =  atoi(optarg);
                break;
            case 'u':
                u = 1;
                break;
            case 'v':
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

    if (B == 1)
        u = 3;

    if (m == 1)
        u = 2;
    dist_type = z;
    bayesian_t *bgame = generate_game(game, a, b, c1, d, C, val, s, n, u);
    polymatrix_t *pgame = bayesian_to_polymatrix(bgame);
    polymatrix_to_file(f, NULL, pgame);
    printf("%d\n", polymatrix_size(pgame));
    bayesian_free(bgame);
    polymatrix_free(pgame);

    return 0;
}
