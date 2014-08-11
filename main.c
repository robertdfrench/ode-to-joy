#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "stepsize.h"
#include "grid.h"
#include "solve_interior.h"
#include "grid_options.h"

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
	
typedef struct timing_measurement_t {
	double beginning;
	double end;
	char* message;
} TimingMeasurement;

TimingMeasurement start_timer(char* message) {
	TimingMeasurement tm;
	tm.message = message;
	tm.beginning = MPI_Wtime();
	tm.end = tm.beginning;
	return tm;
}

#define stop_timer(tm) printf("[Timer] %s: %f seconds\n", tm.message, MPI_Wtime() - tm.beginning); 



void swap_grids(Grid a, Grid b) {
	double* temp = a.internal_storage;
	a.internal_storage = b.internal_storage;
	b.internal_storage = temp;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	GridOptions go = parse_grid_options(argc, argv);

	TimingMeasurement tm = start_timer("Initial Conditions");
	Grid initial_conditions = generate_initial_conditions(go.len_x, go.len_y);
	stop_timer(tm);


	tm = start_timer("Solve Problem");
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
	stop_timer(tm);

	tm = start_timer("Store Grid");
	store_grid(solution_grid);
	stop_timer(tm);
	
	return 0;
}
