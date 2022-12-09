#include <string.h>

#include "lrslib.h"
#include "lrsnash.h"

equilibrium lrs_solve_nash(game * g) {
    equilibrium e = {0};
    e.strats[0].nstrats = g->nstrats[0];
    e.strats[1].nstrats = g->nstrats[1];
    e.strats[0].weights[0] = g->payoff[0][0][0];
    return e;
}