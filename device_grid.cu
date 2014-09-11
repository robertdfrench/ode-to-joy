#include <cuda.h>
extern "C" {
#include "grid.h"
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
