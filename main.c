#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct grid_t {
	double* internal_storage;
	int len_x;
	int len_y;
} Grid;

#define element(i,j) internal_storage[(i * i_len) + (j - 1)]

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
			initial_conditions.element(i,j) = sin(i * hx);
		}
	}
	return initial_conditions;
}

void solve_interior(Grid current, Grid previous) {
} 

#define LEN_X 500
#define LEN_Y 500
#define LEN_T 10

int main(int argc, char** argv) {
	Grid initial_conditions = generate_initial_conditions(LEN_X, LEN_Y);
	double hx = 1.0 / ((double) LEN_X);
	double hy = 1.0 / ((double) LEN_Y);
	double ht = 1.0 / ((double) LEN_T);

	Grid[LEN_T] grids_by_timestep;
	grids_by_timestep[0] = initial_conditions;

	int tau;
	for(tau = 1; tau < LEN_T; tau++) {
		grids_by_timestep[tau] = alloc_grid(LEN_X, LEN_Y);
		solve_interior(grids_by_timestep[tau], grids_by_timestep[tau - 1]);
	}
	
	return 0;
}	
