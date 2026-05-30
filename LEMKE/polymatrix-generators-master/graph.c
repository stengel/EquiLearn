#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "graph.h"

int graph_get_size(char *g_type, int g1, int g2)
{
    if (strncmp(g_type, "Grid", 4) == 0)
        return g1 * g2;
    else if (strncmp(g_type, "Tree", 4) == 0)
        return (pow(g1, g2 + 1) - 1) / (g1 - 1);
    return g1;
}

void graph_fill_char(int **graph, char *g_type, int g1, int g2)
{
    if (strncmp(g_type, "Random", 6) == 0) 
        graph_fill_random(graph, g1, g2);
    else if (strncmp(g_type, "Complete", 8) == 0)
        graph_fill_complete_graph(graph, g1);
    else if (strncmp(g_type, "Cycle", 5) == 0)
        graph_fill_cycle(graph, g1);
    else if (strncmp(g_type, "Grid", 4) == 0)
        graph_fill_grid(graph, g1, g2);
    else if (strncmp(g_type, "Tree", 4) == 0)
        graph_fill_tree(graph, g1, g2);
}

void graph_fill_random(int **graph, int m, int n)
{
    double p = (double) n / 100.0;

    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = i + 1; j < m; ++j) {
            if (((double) rand()) / RAND_MAX <= p) {
                graph[i][j] = 1;
                graph[j][i] = 1;
            }
        }
    }
}

void graph_fill_complete_bipartite(int **graph, int m, int n)
{
    int i, j;

    for (i = 0; i < m; i++)
        //for (j = i + 1; j < n; j++) { //Double-check this is wrong
        for (j = 0; j < n; j++) {
            graph[i][j+m] = 1;
            graph[j+m][i] = 1;
        }
}

void graph_fill_complete_graph(int **graph, int n)
{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = i+1; j < n; j++) {
            graph[i][j] = 1;
            graph[j][i] = 1;
        }
}

void graph_fill_cycle(int **graph, int n)
{
    int i;

    graph[0][1] = 1;
    graph[0][n - 1] = 1;

    for (i = 1; i < n - 1; i++) {
        graph[i][i+1] = 1;
        graph[i][i-1] = 1;
    }

    graph[n - 1][n - 2] = 1;
    graph[n - 1][0] = 1;
}

void graph_fill_grid(int **graph, int m, int n)
{
    int r, c, i, j;

    for (r = 0; r < m; r++) {
        for (c = 0; c < n; c++) {
            i = (r * n) + c;
            if (c < n - 1) {
                j = (r * n) + c + 1;
                graph[i][j] = 1;
                graph[j][i] = 1;
            }
            if (r < m - 1) {
                j = ((r + 1) * n) + c;
                graph[i][j] = 1;
                graph[j][i] = 1;
            }
        }
    }
}

void graph_fill_tree(int **graph, int b, int d)
{
    int m = (pow(b, d) - 1) / (b - 1);

    int i;
    for (i = 0; i < m; ++i) {
        int j;
        for (j = (b*i + 1); j <= (b*i + b); ++j) {
            graph[i][j] = 1;
            graph[j][i] = 1;
        }
    }
}
