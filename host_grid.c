#include <stdio.h>
#include <stdlib.h>
#include "grid.h"

void OTJ_Grid_Swap(OTJ_Grid* a, OTJ_Grid* b) {
	double* temp = a->internal_storage;
	a->internal_storage = b->internal_storage;
	b->internal_storage = temp;
}

OTJ_Grid OTJ_Grid_Alloc(int len_x, int len_y) {
	OTJ_Grid g;
	g.len_x = len_x;
	g.len_y = len_y;
	g.internal_storage = (double*)malloc(sizeof(double) * len_x * len_y);
	return g;
}

void OTJ_Grid_Store(OTJ_Grid g) {
	FILE* gridFile = fopen("output.otj_grid","w");
	size_t num_cells = g.len_x * g.len_y;
	size_t cell_size = sizeof(double);
	size_t elements_written = fwrite(g.internal_storage, cell_size, num_cells, gridFile);
	if (elements_written < num_cells) {
		printf("An error occurred while saving the grid.\n");
	}
}
