#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "stepsize.h"
#include "grid.h"
#include "solve_interior.h"
#include "grid_options.h"
#include "timing_measurement.h"

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

void swap_grids(Grid a, Grid b) {
	double* temp = a.internal_storage;
	a.internal_storage = b.internal_storage;
	b.internal_storage = temp;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	GridOptions go = parse_grid_options(argc, argv);

	OTJ_Timer tm = OTJ_Timer_Start("Initial Conditions");
	Grid initial_conditions = generate_initial_conditions(go.len_x, go.len_y);
	OTJ_Timer_Stop(tm);


	tm = OTJ_Timer_Start("Solve Problem");
	Stepsize h = stepsize_from_grid_options(go);

	Grid current_grid = createDeviceGrid(initial_conditions);
	Grid previous_grid = createAndCopyDeviceGrid(initial_conditions);
	swap_grids(current_grid, previous_grid);
	int tau;
	for(tau = 1; tau < go.len_t; tau++) {
		swap_grids(current_grid, previous_grid);
		apply_boundary_conditions(current_grid);
		solve_interior(current_grid, previous_grid,h);
	}
	Grid solution_grid = alloc_grid(go.len_x, go.len_y);
	retrieveDeviceGrid(solution_grid, current_grid);
	OTJ_Timer_Stop(tm);

	tm = OTJ_Timer_Start("Store Grid");
	store_grid(solution_grid);
	OTJ_Timer_Stop(tm);
	
	return 0;
}
