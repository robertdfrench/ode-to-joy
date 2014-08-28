#include <stdio.h>
#include <stdlib.h>
#include "grid.h"
void OTJ_Grid_Store(OTJ_Grid g) {
	FILE* gridFile = fopen("output.otj_grid","w");
	size_t num_cells = g.len_x * g.len_y;
	size_t cell_size = sizeof(double);
	size_t elements_written = fwrite(g.internal_storage, cell_size, num_cells, gridFile);
	if (elements_written < num_cells) {
		printf("An error occurred while saving the grid.\n");
	}
}

OTJ_Grid OTJ_Grid_Alloc(int len_x, int len_y) {
	OTJ_Grid g;
	g.len_x = len_x;
	g.len_y = len_y;
	g.internal_storage = (double*)malloc(sizeof(double) * len_x * len_y);
	return g;
}

extern "C" OTJ_Grid createDeviceGrid(OTJ_Grid host_grid) {
	OTJ_Grid device_grid = host_grid;
	cudaMalloc(&device_grid.internal_storage, OTJ_Grid_Size(host_grid));
	return device_grid;
}

extern "C" OTJ_Grid createAndCopyDeviceGrid(OTJ_Grid host_grid) {
	OTJ_Grid device_grid = createDeviceGrid(host_grid);
	cudaMemcpy(device_grid.internal_storage, host_grid.internal_storage, OTJ_Grid_Size(device_grid), cudaMemcpyHostToDevice);
	return device_grid;
}

extern "C" void retrieveDeviceGrid(OTJ_Grid host_grid, OTJ_Grid device_grid) {
	cudaMemcpy(host_grid.internal_storage, device_grid.internal_storage, OTJ_Grid_Size(device_grid), cudaMemcpyDeviceToHost);
}
