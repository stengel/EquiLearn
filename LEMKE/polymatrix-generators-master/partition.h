#ifndef PARTITION_H
#define PARTITION_H

extern int **partitions;

int **get_partitions();
void partition_free(int **part, int count);
void partition_func(int n, int k, int l, int d, int *arr, int *perm);
int sorted_partitions(int n, int k);
#endif
