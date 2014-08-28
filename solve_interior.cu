#include <cuda.h>
#include "stepsize.h"
#include "grid.h"

__global__ void north_boundary(OTJ_Grid g) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < g.len_x) OTJ_Grid_Element(g,i,0) = 0.0;
}

__global__ void south_boundary(OTJ_Grid g) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < g.len_x) OTJ_Grid_Element(g,i,g.len_y - 1) = 0.0;
}

__global__ void east_boundary(OTJ_Grid g) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < g.len_y) OTJ_Grid_Element(g,0,j) = 0.0;
}

__global__ void west_boundary(OTJ_Grid g) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < g.len_y) OTJ_Grid_Element(g,g.len_x - 1,j) = 0.0;
}

__global__ void solve_interior_cell(OTJ_Grid previous, OTJ_Grid current, Stepsize h) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	
	if(i < current.len_x - 1 && j < current.len_y - 1) { 
		double uijn = OTJ_Grid_Element(previous,i,j);
		double uiP1jn = OTJ_Grid_Element(previous,i+1,j);
		double uiM1jn = OTJ_Grid_Element(previous,i-1,j);
		double uijP1n = OTJ_Grid_Element(previous,i,j+1);
		double uijM1n = OTJ_Grid_Element(previous,i,j-1);

		double t_contribution = uijn;
		double x_contribution = h.t * (uiP1jn - 2*uijn + uiM1jn) / (h.x * h.x);
		double y_contribution = h.t * (uijP1n - 2*uijn + uijM1n) / (h.y * h.y);

		double uijnP1 = t_contribution + x_contribution + y_contribution;
		OTJ_Grid_Element(current,i,j) = uijnP1;	
	}
}

extern "C" void solve_interior(OTJ_Grid current, OTJ_Grid previous, Stepsize h) {
	int max_i = current.len_x - 2;
	int max_j = current.len_y - 2;
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(max_i / 32 + 1, max_j / 32 + 1);

	solve_interior_cell<<<numBlocks,threadsPerBlock>>>(previous, current, h);
	cudaDeviceSynchronize();

}



extern "C" void apply_boundary_conditions(OTJ_Grid g) {
	int max_i = g.len_x - 2;
	int max_j = g.len_y - 2;
	dim3 threadsPerBlock(1024);
	dim3 numHorizontalBlocks(max_i / 1024 + 1);
	dim3 numVerticalBlocks(max_j / 1024 + 1);

	north_boundary<<<numHorizontalBlocks,threadsPerBlock>>>(g);
	south_boundary<<<numHorizontalBlocks,threadsPerBlock>>>(g);
	east_boundary<<<numVerticalBlocks,threadsPerBlock>>>(g);
	west_boundary<<<numVerticalBlocks,threadsPerBlock>>>(g);
}
