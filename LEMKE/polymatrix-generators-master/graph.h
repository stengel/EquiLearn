#ifndef GRAPH_H
#define GRAPH_H

int graph_get_size(char *g_type, int g1, int g2);
void graph_fill_complete_bipartite(int **graph, int m, int n);
void graph_fill_random(int **graph, int m, int n);
void graph_fill_complete_graph(int **graph, int n);
void graph_fill_cycle(int **graph, int n);
void graph_fill_grid(int **graph, int m, int n);
void graph_fill_tree(int **graph, int b, int d);
void graph_fill_char(int **graph, char *g_type, int g1, int g2);

#endif
