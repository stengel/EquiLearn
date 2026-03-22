---
title: Exact-Arithmetic Implementation of Lemke's Algorithm
author: Bernhard von Stengel 
date: 22 March 2026
---

We document the exact-arithmetic implementation of
Lemke's algorithm as in
<https://github.com/stengel/EquiLearn/tree/main/LEMKE/>

It has been moved to GAMBIT with very minor modifications
(such as removed first lines `#!/usr/bin/python`) at
<https://github.com/gambitproject/lemke/tree/main/src>.

The mathematical background is described in
<https://github.com/stengel/EquiLearn/tree/main/LEMKE/TEX-DOCS>
(which includes this documentation) in PDF files, many of
them with .tex and auxiliary sources, including

- `using-lemke.pdf` that briefly explains Lemke's algorithm
  and its application to the *tracing procedure* and the
  *Lemke-Howson algorithm* (the global Newton method too but
  not used here) as used in `bimatrix.py`.

- `how-randomstart.pdf` describes how to generate uniformly
  distributed points from the unit simplex, and rounding
  them to a prescribed limited accuracy (the dominator for
  rational numbers, which can be as low as 1).

- `dhivya-chap2.pdf` (PDF only) is chapter
  2 of the PhD dissertation (as per 22 May 2026 submitted to
  the University of Glasgow) of Dhivya Anand Kumar that
  describes the tracing procedure for *polymatrix games*
  and backwards-forwards search.

# Current implementation of Lemke's algorithm

There are 2 main versions:

- an original version in C using GMP GNU Multi-Precision
  Arithmetic,

- a ported Python version, using the implicit arbitrary
  precision of Python integers. Important: all arrays are
  handled as lists of lists and *never* via `numpy`, which
  would destroy the arbitrary precision.

The main features of the implementation are:

- arbitrary precision with integer pivoting (much faster
  than using rational arithmetic),

- consistent handling of *degeneracy* and corresponding
  careful initialization (which for games always has initial
  degenerate steps even if the game itself nondegenerate),

- storage only of a reduced tableau (rather than a full
  tableau where basic variables are stored alongside their
  unit columns).

## Integer pivoting

Given rational inputs, all columns of the LCP are scaled to 
integers, with the corresponding scale factors remembered,
after which a solution is translated back to the original
meaning of the columns.
Example: If the column for variable $z_i$ is multiplied
by 10, this means that the new variable $z_i'$ for the new
system has $1/10$ of the value that $z_i$ would have in a
solution.
Hence, the computed value of $z_i'$ needs to be multiplied
by the scaling factor 10 to represent the actual value of
$z_i$.

*Pivoting* means that a nonbasic variable (called *cobasic*
following the terminology of *lrs*) becomes basic (this is
the entering variable) and replaces a basic variable (called
the leaving variable). 
In a standard full tableau that lists all the variables with
their corresponding columns, basic variables have unit
vectors as columns.
In pivoting, a nonbasic column is transformed via row
operations into a basic column, where

- the pivot element, in the column for the entering variable
  and the row for the leaving variable, must be *nonzero*.
  Except at the very beginning when the system is
  infeasible, the pivot element is *positive* to preserve
  feasibility of the basic feasible solution.

- standard pivoting turns the entering column into a unit
  column, via 

  - dividing the pivot row by the pivot element,

  - substracting suitable multiples of the pivot row to 
    create zero elements elsewhere in the pivot column.

- *integer* pivoting proceeds differently:

  - every other row is *multiplied* by the pivot element,

  - a corresponding multiple of the pivot row is subtracted
    to create zero entries in the pivot column,

  - and now the main step, to prevent growth of the integers
    in the tableau: all other rows have the property that
    all entries are *multiples of the previous pivot
    element*, which can thus be divided out without
    remainder;

  - this works because the pivot element is the
    *determinant* of the basic columns, with the solution
    given by Cramer's rule;

  - the whole thing is the reason why exact Gaussian
    elimination works in polynomial time.

What needs to be stored?

- In a reduced tableau, only the right-hand side and the
  nonbasic columns, plus
  (in the code via the arrays `bascobas` and `whichvar`)

    - which (cobasic) variable corresponds to the respective
      nonbasic column,

    - which (basic) variable corresponds to a row,

    - the determinant of the matrix of the basic columns, a single number 
      that is always the last pivot element.

## Handling degeneracy

Degeneracy is a non-uniqueness of the leaving variable,
which would throw the algorithm if not consistently handled.

This is done by *lexicographic perturbation*:

- in the system $Iw-dz_0-Mz=q$ (where $I$ is the identity matrix),
  pretend that the right-hand side $q$ is replaced by
  $q(\varepsilon)=q+(\varepsilon,\varepsilon^2,\ldots,\varepsilon^n)^{\top}$
  for sufficiently small positive $\varepsilon$.

- during the computation any full tableau is the 
  system $Iw-dz_0-Mz=q$ multiplied by the inverse $A_B^{-1}$
  of the current basis matrix. 

  -   The $n$ columns of that inverse are the coefficients of 
      $\varepsilon,\varepsilon^2,\ldots,\varepsilon^n$ on the
      right-hand side.

  -   We maintain a *positive* right-hand side
      $A_B^{-1}q(\varepsilon)$, which is equivalent to the
      matrix
      $[A_B^{-1}q~~A_B^{-1}]$ being *lexico-positive*
      (the first non-zero entry of each row is positive).

  -   The columns of $A_B^{-1}$ can be identified from the
      reduced tableau as the columns of $w_1,\ldots,w_n$,
      which are either unit vectors if $w_i$ is basic
      (scaling does not matter for lexico-positivity)
      or the respective nonbasic column if $w_i$ is
      nonbasic.

  - The lexico-positivity is maintained via a
    *lexico-minimum ratio test*. 

Special care is needed for *initialization*.
The initial solution $w=q$ is infeasible if $q$ has negative
components. 
For the system $w=q+dz_0$ (which constitutes the *primary
ray* in Lemke's algorithm for all sufficiently large $z_0$),
we get a first lexico-positive solution via the
lexico-minimum ratio test applied to a negated tableau
column.
The *covering vector* $d$ is of the form that the first 
pivots are usually degenerate in the game-theoretic
applications, so this is particularly important.

# Improvements to Lemke

The program is currently

- a standalone version to process an LCP from a file,

- a library of methods that can be called, in particular for
  solving games.

## More pythonic

The program was ported from C to Python.

The only bug that was introduced (subsequently fixed)
was in the lexico-minimum ratio test when the set of
tied rows for the leaving variable was converted to 
a Python version to replace the array-style C version.

This demonstrates that any change to make the code more pythonic
(essentially for readability) must be accompagnied by
constant tests that functionality is not broken.

## Test suites

We need a set of systematic tests that explore all decision
paths in the tree (for degeneracy, ray termination, other
edge cases).

These should have a number of test inputs alongside a
program that is known to work.

## Better switches to control outputs

The program has a number of switches to control its
behaviour, in particular its output.
In the original C program they were packaged in a `struct`,
essentially a record with individual fields, that was then
passed to the algorithm (which therefore did not need an
amended list of arguments in case new switches were to be
introduced).
This looked as follows, with `int` representing Boolean
values if the variable names start with '`b`':

    /* flags for  runlemke  */
    typedef struct
        {
        int   maxcount  ;   /* max no. of iterations, infinity if 0         */
        int   bdocupivot;   /* Y/N  document pivot step                     */
        int   binitabl ;    /* Y/N  output entire tableau at beginning/end  */
        int   bouttabl  ;   /* Y/N  output entire tableau at each step      */
        int   boutsol   ;   /* Y/N  output solution                         */
        int   binteract ;   /* Y/N  interactive pivoting                    */
        int   blexstats ;   /* Y/N  statistics on lexminratio tests         */
        }
        Flagsrunlemke;

These are currently set as global variables in the Python
program, which should be changed.

There are two conceivable ways of changing this:

- introducing an own class for these parameters:

- the corresponding class object would also be modified by
a more systematic way of processing command-line
arguments with the `Click` package if the program is called stand-alone;
this should also be done by other programs of this sort,
such as processing games as discussed below.

- introducing long parameter lists, much of which have
default values and would only be modified if necessary,
syntactically by explicit assignments in the call such
that the parameter order does not matter.

The first seems more compact, while the second is more
explicit. 
The first seems better.

## More fine-grained modularity

When the program is used as a library, it may make sense
to use the tableau (via a suitable switch) also without the
covering vector $d$ with a different initialization like in
the Lemke-Howson algorithm.

Similarly, another initialization may mean to start with an
existing LCP solution (and corresponding tableaux) as a
``backwards'' run of the algorithm.
(This may result in ray termination with the primary ray,
which needs to be caught.)

In both cases, the first entering variable has to be
specified.

## Using existing utility packages

There are two small packages for `lemke.py`, which should be
studied and kept but not duplicated:

-   `columnprint.py` is a class to print outputs like tableaus
    and matrices such that output uses as few whitespace between
    columns as possible.
    That is, each column is of optimal (smallest) width and not
    fixed-width.
    This is extremely useful for viewing tabular outputs on screen.

-   `utils.py` has a small set of routines for reading 
    from files, for creating vectors and matrices as lists,
    and for creating fractions with prescribed
    accuracy of decimals.
    In particular, it does not use the unexpected rounding
    function of Python, which rounds 1.5 to 2 and 2.5 to 2
    (called "banker's rounding" that "rounds" .5 *up or down*
    to the nearest *even* integer so that rounding errors,
    presumably of half-pennies, tend to cancel out).
    Rounding applies to the absolute value, with .5 always
    rounded up, and then taking the proper sign.

# Applications to solving games: Current state

## Bimatrix games: The tracing procedure

The main application of `lemke.py` is to finding Nash
equilibria of bimatrix games, currently implemented in 
`bimatrix.py`. 
Besides Lemke, it imports another auxiliary module called 
`randomstart.py` (see below).

Nash equilibria of a bimatrix game are solutions to an LCP
with the two payoff matrices as input.
The LCP variables are the mixed-strategy probabilities and
two payoff variables.

The payoff variables have arbitrary sign.
However, standard LCP variables have nonnegative variables
only. 
In order to avoid the complication of representing the
payoff variables as differences of two nonnegative
variables, the standard LCP solution can be used by
guaranteeing *negative payoffs* by making all matrix
payoffs negative by subtracting a sufficiently large
constant integer from each matrix.

Correspondingly, the equality for the mixed-strategy
probabilities can then be represented by a single
*inequality* that these probabilities sum to *at least* 1. 

The main specialty of using Lemke is by mimicking the
*tracing procedure* (the linear tracing procedure by
Harsanyi and Selten) with a choice of the covering vector
$d$ in Lemke's algorithm.
That covering vector is obtained by multiplying
each payoff matrix with a fixed mixed strategy of the other player
called the *prior*.

The resulting iterations of Lemke then correspond to
equilibria of an intermittent game where the players
play with probability $z_0$ against the prior and with the
complementary probability against their actual strategies.
The path traversed by these steps is a "homotopy path"
of these parametrised games.
Once $z_0$ leaves the basis (which is when Lemke
terminates), an actual equilibrium is found.

If the prior is completely mixed, the equilibrium found is also
*perfect* in the sense of Selten's trembling-hand perfection.

Initialization always requires a few degenerate pivoting
steps to pivot the payoff variables into the basis, which
then stay basic throughout the computation.

The tracing procedure is implemented with the possibility of
starting with uniform probabilities as the "standard prior",
but also with *random* priors.
These random priors are chosen uniformly from the simplex of
mixed strategies (which is not completely trivial).
In order to keep the entries of the covering vector as
reasonably small integers, the accuracy of the probabilities
is *limited*, e.g. to a maximal denominator of 1,000 or 10,000
(which can be chosen). The random probabilities are then
normalized to sum to the denominator
(see separate documentation on `randomstart.py`). 

Finding equilibria via the tracing procedure with varying
random starting points provides a kind of equilibrium
selection method.
The equilibria are usually found with numbers of pivots
roughly proportional to the LCP dimension.
(No apparent issue of computational intractability due to
PPAD-hardness.)

## Bimatrix games: Mimicking Lemke-Howson

Lemke's algorithm can also mimick the Lemke-Howson procedure
by setting the covering vector to all-1 entries, except for
a 0 for the *missing label*. 

There is an alternative representation of Lemke-Howson
which *omits* the payoff variables and instead uses
unscaled nonnegative variables for the mixed strategies.
The payoff is normalized to 1, and is re-scaled alongside
the mixed-strategy variables when they become probabilities
again.
For this purpose, all payoffs have to be *positive* (or at
least nonnegative with no zero column in the LCP).
The algorithm starts with an *existing* trivial (all-zero)
LCP solution called the *artificial equilibrium*.
It is not currently implemented, and may be useful
for comparison purposes.

## Two-player games: The sequence form

Lemke-Howson cannot be easily extended to the *sequence
form*, a compact strategic description for an extensive game
with perfect recall,
but the tracing procedure with Lemke can.

The sequence form uses multiple payoff variables with
corresponding constraints. 
It has been used with the C version of Lemke's algorithm
for a specific class of games.
The extensive form of that game was general, but
in C and without a method that generates this data structure
from a representation of the extensive game such as a game
in `.ef` or `.efg` format.

## Polymatrix games: The tracing procedure

Polymatrix games are $n$-player games composed
of sums of pairwise interactions.
Each player chooses a pure strategy, simultaneously
interacts with each other player via a payoff matrix,
and gets the sum of the respective payoffs as a reward.
This can be converted to an LCP.

The LCP can be solved by Lemke via the tracing procedure,
via a prior mixed strategy for each player as a starting
point. 
This generalizes the bimatrix case.
It has been implemented and has the tracing procedures for
a bimatrix game as a special case,
see
<https://github.com/stengel/EquiLearn/blob/main/LEMKE/polymatrix.py>
It is based on the older implementation of Lemke and
`bimatrix.py` in the same repository.

## Tracing backwards

The tracing procedure (and the Lemke-Howson algorithm) only
finds equilibria of positive index (or of that positive index
after a lexicographic perturbation if the game is
degenerate).

If two different equilibria are found, say $z$ and $z'$,
these are LCP solutions.
In the final tableau for $z$ one can then use the covering
vector $d'$ that led to $z'$, replace the corresponding
column for $z_0$ in the tableau (this requires multiplying 
$d'$ with the current inverse of the basis matrix), and
run Lemke *backwards* by now letting $z_0$ enter the basis.
This cannot lead back along the path to the starting primary
ray that led to $z$, because the path is parametrized by
$d'$, which leads to~$z'$.
Hence, the backwards trace leads to a third equilibrium,
of negative index.

Care has to be taken if $z$ is degenerate in that the
final tableau that represents $z$ has to be lexico-positive
(this property is independent of the covering vector).
That is, in a degenerate situation, using the lexico-minimum
ratio test takes precedence over the possibility that $z_0$
can leave the basis, which would speed up the computation
in that situation (this is controlled by a -- possibly
currently disabled -- switch in Lemke, to let $z_0$ leave
immediately if possible or not).

It is possible to generate a sequence of backwards-forwards
runs of Lemke to generate a whole network of equilibria.
It may be an alternative to finding as many equilibria as
possible compared to *lrsNash*.

This backwards-forwards search has been implemented for
polymatrix (and consequently also bimatrix) games.
It has not been thoroughly tested, nor fully analyzed
theoretically, for bimatrix games, where the concept of
index needs to be defined more carefully with the
lexicographic perturbation. In particular, one and the
same equilibrium may have a different index depending
on its choice of basic variables; the index may have to 
be more properly defined for the basis rather than the
equilibrium.

# Applications to solving games: Improvements

The main improvement is an improved control of switches for
the Lemke program via its library functions.

Similarly, the switches for the bimatrix or polymatrix
solver need to be structured more systematically, as well
their control via command-line parameters
(or specifications in the input file).





