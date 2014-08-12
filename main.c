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

OTJ_Grid generate_initial_conditions(int len_x, int len_y) {
	OTJ_Grid initial_conditions = OTJ_Grid_Alloc(len_x, len_y);
	int i,j;
	double hx = 1.0 / ((double) len_x);
	for (i = 0; i < len_x; i++) {
		for (j = 0; j < len_y; j++) {
			OTJ_Grid_Element(initial_conditions,i,j) = sin(i * hx);
		}
	}
	return initial_conditions;
}

void swap_grids(OTJ_Grid a, OTJ_Grid b) {
	double* temp = a.internal_storage;
	a.internal_storage = b.internal_storage;
	b.internal_storage = temp;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	GridOptions go = parse_grid_options(argc, argv);

	OTJ_Timer tm = OTJ_Timer_Start("Initial Conditions");
	OTJ_Grid initial_conditions = generate_initial_conditions(go.len_x, go.len_y);
	OTJ_Timer_Stop(tm);


	tm = OTJ_Timer_Start("Solve Problem");
	Stepsize h = stepsize_from_grid_options(go);

	OTJ_Grid current_grid = createDeviceGrid(initial_conditions);
	OTJ_Grid previous_grid = createAndCopyDeviceGrid(initial_conditions);
	swap_grids(current_grid, previous_grid);
	int tau;
	for(tau = 1; tau < go.len_t; tau++) {
		swap_grids(current_grid, previous_grid);
		apply_boundary_conditions(current_grid);
		solve_interior(current_grid, previous_grid,h);
	}
	OTJ_Grid solution_grid = OTJ_Grid_Alloc(go.len_x, go.len_y);
	retrieveDeviceGrid(solution_grid, current_grid);
	OTJ_Timer_Stop(tm);

	tm = OTJ_Timer_Start("Store Grid");
	OTJ_Grid_Store(solution_grid);
	OTJ_Timer_Stop(tm);
	
	return 0;
}
