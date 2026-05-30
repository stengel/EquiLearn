#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "util.h"
#include "matrix.h"

/*
Uniformly random int from range [0, n).
Extract from 
http://stackoverflow.com/questions/822323/how-to-generate-a-random-number-in-c
*/
int randint(int n)
{
    if ((n - 1) == RAND_MAX) {
        return rand();
    } else {
        // Chop off all of the values that would cause skew...
        long end = RAND_MAX / n; // truncate skew
        assert (end > 0L);
        end *= n;

        // ... and ignore results from rand() that fall above that limit.
        // (Worst case the loop condition should succeed 50% of the time,
        // so we can expect to bail out of this loop pretty quickly.)
        int r;
        while ((r = rand()) >= end);

        return r % n;
    }
}

/*
 * Generates a random number from a normal distribution
 * http://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
 */
double rand_norm (double mu, double sigma)
{
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;

    if (call)
    {
        call = !call;
        return (mu + sigma * (double) X2);
    }

    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double) X1);
}

/* 
 * Given the cholesky decomposition A of the covariance matrix, generate a
 * vector from a multivariate normal distribution
 */ 
static matrix_t *rand_multivariate_cholesky(matrix_t *A, double mu, matrix_t *z)
{
    matrix_rand_norm(z, 0, 1);
    matrix_t *t1 = matrix_mul_mat_vec(A, z);
    matrix_t *x = matrix_add_const(t1, mu);
    return x;
}

/*
 * Given a covariant matrix 'cov' of size k x k, and the expected average mu,
 * generates a random vector which is k-variate normally distributed.
 */
matrix_t *rand_multivariate_single(matrix_t *cov, double mu)
{
    matrix_t *A = matrix_cholesky(cov);
    matrix_t *z = matrix_alloc(cov->nrows, 1);
    matrix_t *x = rand_multivariate_cholesky(A, mu, z);
    matrix_free(A);
    matrix_free(z);
    return x;
}

/*
 * Given a covariant matrix 'cov' of size k x k, and the expected average mu,
 * generates a matrix of size reps x k where each row is chosen from a k-variate
 * normal distribution using the specified covariance and mean.
 */
matrix_t *rand_multivariate_multiple(matrix_t *cov, double mu, int reps)
{
    matrix_t *A = matrix_cholesky(cov);
    matrix_t *res = matrix_alloc(reps, cov->nrows);
    matrix_t *z = matrix_alloc(cov->nrows, 1);
    int i;
    for (i = 0; i < reps; ++i){
        matrix_t *x = rand_multivariate_cholesky(A, mu, z);
        int j;
        for (j = 0; j < x->nrows; ++j)
            res->data[i][j] = x->data[j][0];
        matrix_free(x);
    }
    matrix_free(A);
    matrix_free(z);
    return res;
}

int *init_subset(int *s, int n)
{
    int *sub = calloc(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i)
        if (s[i]) {
            sub[i] = s[i];
            break;
        }

    return sub;
}

int next_subset(int *s, int *sub, int n)
{
    int i, carry = 1;

    for (i = 0; i < n; ++i) {
        if (sub[i] == s[i]) {
            sub[i] = 0;
        } else {
            carry = 0;
            sub[i] = s[i];
            break;
        }
    }

    if (carry)
        return 0;

    return 1;
}

int myRandom (int size) {
    int i, n;
    static int numNums = 0;
    static int *numArr = NULL;

    // Initialize with a specific size.

    if (size >= 0) {
        if (numArr != NULL)
            free (numArr);
        if ((numArr = malloc (sizeof(int) * size)) == NULL)
            return -1;
        for (i = 0; i  < size; i++)
            numArr[i] = i;
        numNums = size;
    }

    // Error if no numbers left in pool.

    if (numNums == 0)
       return -2;

    // Get random number from pool and remove it (rnd in this
    //   case returns a number between 0 and numNums-1 inclusive).

    n = rand() % numNums;
    i = numArr[n];
    numArr[n] = numArr[numNums-1];
    numNums--;
    if (numNums == 0) {
        free (numArr);
        numArr = 0;
    }

    return i;
}

void print_int(int *arr, int n)
{
    int i;
    printf("(");
    for (i = 0; i < n; ++i) {
        printf("%d", arr[i]);
        if (i < n-1)
            printf(", ");
    }

    printf(")\n");
}

void print_double(double *v, int n)
{
    int j;
    for (j = 0; j < n; j++)
        printf("%lf ", v[j]);
    printf("\n");
}

int find_draws(int *wins, int k)
{
    int i;

    int count = 0;
    for (i = 0; i < k; ++i)
        if (wins[i] == 2)
            count++;

    return count;
}
