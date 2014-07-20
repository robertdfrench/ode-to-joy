#include "grid.h"
#include <stdio.h>
void store_grid(Grid g) {
	FILE* gridFile = fopen("output.otj_grid","w");
	size_t num_cells = g.len_x * g.len_y;
	size_t cell_size = sizeof(double);
	size_t elements_written = fwrite(g.internal_storage, cell_size, num_cells, gridFile);
	if (elements_written < num_cells) {
		printf("An error occurred while saving the grid.\n");
	}
}
