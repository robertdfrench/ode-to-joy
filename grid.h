#ifndef OTJ_GRID_HEADER
#define OTJ_GRID_HEADER 1
typedef struct grid_t {
	double* internal_storage;
	int len_x;
	int len_y;
} Grid;

#define grid_size(g) sizeof(double) * g.len_x * g.len_y
#define grid_element(g,i,j) g.internal_storage[(i * g.len_x) + (j - 1)]

void store_grid(Grid g);
#endif
