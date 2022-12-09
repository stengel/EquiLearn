
#define MAXSTRAT 200
#define ROW 0
#define COL 1

typedef struct {
        long num;
        long den;
} ratnum;

typedef struct {
  long nstrats[2];
  ratnum payoff[MAXSTRAT][MAXSTRAT][2];
  //might need to include aux
} game;

typedef struct {
    long nstrats;
    ratnum weights[MAXSTRAT];
} mixedstrat;

typedef struct {
  mixedstrat strats[2];
} equilibrium; //TODO: figure out linked list in cython

equilibrium lrs_solve_nash(game *g);