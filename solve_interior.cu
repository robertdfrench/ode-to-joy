#include <cuda.h>
#include "grid.h"
#include "stepsize.h"

__global__ void solve_interior_cell(Grid previous, Grid current, Stepsize h) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	
	if(i < current.len_x - 1 && j < current.len_y - 1) { 
		float uijn = grid_element(previous,i,j);
		float uiP1jn = grid_element(previous,i+1,j);
		float uiM1jn = grid_element(previous,i-1,j);
		float uijP1n = grid_element(previous,i,j+1);
		float uijM1n = grid_element(previous,i,j-1);

		float t_contribution = uijn;
		float x_contribution = h.t * (uiP1jn - 2*uijn + uiM1jn) / (h.x * h.x);
		float y_contribution = h.t * (uijP1n - 2*uijn + uijM1n) / (h.y * h.y);

		float uijnP1 = t_contribution + x_contribution + y_contribution;
		grid_element(current,i,j) = uijnP1;	
	}
}

Grid createDeviceGrid(Grid host_grid) {
	Grid device_grid = host_grid;
	cudaMalloc(&device_grid.internal_storage, grid_size(host_grid));
	return device_grid;
}

extern "C" void solve_interior(Grid current, Grid previous, Stepsize h) {
	int max_i = current.len_x - 2;
	int max_j = current.len_y - 2;
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(max_i / 32 + 1, max_j / 32 + 1);

	Grid device_current = createDeviceGrid(current);
	Grid device_previous = createDeviceGrid(previous);
	cudaMemcpy(device_previous.internal_storage, previous.internal_storage, grid_size(previous), cudaMemcpyHostToDevice);

	solve_interior_cell<<<numBlocks,threadsPerBlock>>>(device_previous, device_current, h);
	cudaDeviceSynchronize();

	cudaMemcpy(current.internal_storage, device_current.internal_storage, grid_size(current), cudaMemcpyDeviceToHost);
	cudaFree(device_current.internal_storage);
	cudaFree(device_previous.internal_storage);
} 
