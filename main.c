#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct stepsize_t {
	double x;
	double y;
	double t;
} Stepsize;

typedef struct grid_t {
	double* internal_storage;
	int len_x;
	int len_y;
} Grid;

#define grid_element(g,i,j) g.internal_storage[(i * g.len_x) + (j - 1)]

Grid alloc_grid(int len_x, int len_y) {
	Grid g;
	g.len_x = len_x;
	g.len_y = len_y;
	g.internal_storage = (double*)malloc(sizeof(double) * len_x * len_y);
	return g;
}

Grid generate_initial_conditions(int len_x, int len_y) {
	Grid initial_conditions = alloc_grid(len_x, len_y);
	int i,j;
	double hx = 1.0 / ((double) len_x);
	for (i = 0; i < len_x; i++) {
		for (j = 0; j < len_y; j++) {
			grid_element(initial_conditions,i,j) = sin(i * hx);
		}
	}
	return initial_conditions;
}

void solve_interior(Grid current, Grid previous, Stepsize h) {
	int i,j;
	int max_i = current.len_x - 2;
	int max_j = current.len_y - 2;
	for(i = 1; i < max_i; i++) {
		for(j = 1; j < max_j; j++) {
			double uijn = grid_element( previous,i,j);
			double t_contribution = uijn;

			double uiP1jn = grid_element( previous,i+1,j);
			double uiM1jn = grid_element(previous,i-1,j);
			double x_contribution = h.t * (uiP1jn - 2*uijn + uiM1jn) / (h.x * h.x);

			double uijP1n = grid_element(previous,i,j+1);
			double uijM1n = grid_element(previous,i,j-1);
			double y_contribution = h.t * (uijP1n - 2*uijn + uijM1n) / (h.y * h.y);

			double uijnP1 = t_contribution + x_contribution + y_contribution;
			grid_element(current,i,j) = uijnP1;	
		}
	}
} 

void apply_boundary_conditions(Grid g) {
	int i,j;
	// North Boundary
	for(i=0; i < g.len_x; i++) grid_element(g,i,0) = 0;

	// South Boundary
	for(i=0; i < g.len_x; i++) grid_element(g,i,g.len_y - 1) = 0;

	// East Boundary
	for(j=0; j < g.len_y; j++) grid_element(g,0,j) = 0;

	// West Boundary
	for(j=0; j < g.len_y; j++) grid_element(g,g.len_x - 1,j) = 0;
}

#define LEN_X 500
#define LEN_Y 500
#define LEN_T 10

void store_grid(Grid g) {
	FILE* gridFile = fopen("output.otj_grid","w");
	size_t num_cells = g.len_x * g.len_y;
	size_t cell_size = sizeof(double);
	size_t elements_written = fwrite(g.internal_storage, cell_size, num_cells, gridFile);
	if (elements_written < num_cells) {
		printf("An error occurred while saving the grid.\n");
	}
}
	

int main(int argc, char** argv) {
	Grid initial_conditions = generate_initial_conditions(LEN_X, LEN_Y);
	Stepsize h;
	h.x = 1.0 / ((double) LEN_X);
	h.y = 1.0 / ((double) LEN_Y);
	h.t = 1.0 / ((double) LEN_T);

	Grid grids_by_timestep[LEN_T];
	grids_by_timestep[0] = initial_conditions;

	int tau;
	for(tau = 1; tau < LEN_T; tau++) {
		grids_by_timestep[tau] = alloc_grid(LEN_X, LEN_Y);
		apply_boundary_conditions(grids_by_timestep[tau]);
		solve_interior(grids_by_timestep[tau], grids_by_timestep[tau - 1],h);
	}

	store_grid(grids_by_timestep[LEN_T - 1]);
	
	return 0;
}
