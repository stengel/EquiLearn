please read ../TeX/usinglemke.pdf

# Starting the Python programs:

for help try

python bimatrix.py -?

python lemke.py -?

python randomstart.py

# Bimatrix

This is the important program.
It can be used from a file, specified like the input for lrsnash.

The default filename is "game".

## Example uses:

`python bimatrix.py -LH`

computes all Lemke-Howson paths.

`python bimatrix.py 6x6game -trace 1000`

traces the equilibria of the game in `6x6game` (which has 75
equilibria) for 1000 random starting priors.
