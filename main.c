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


double analytic_solution(double x, double y, double t) {
	int infinity = 1;
	int r;
	double sum = 0.0;
	for(r = 1; r < infinity; r++) {
		sum += (1 - pow(-1,r)) * sin(0.5 * r * M_PI * x) * pow(M_E,-0.25 * M_PI * M_PI * (r * r + 1) * t);
	}
	return sin(0.5 * M_PI * y) * sum;
}

OTJ_Grid generate_initial_conditions(int len_x, int len_y) {
	OTJ_Grid initial_conditions = OTJ_Grid_Alloc(len_x, len_y);
	int i,j;
	double hx = 1.0 / ((double) len_x);
	for (i = 0; i < len_x; i++) {
		for (j = 0; j < len_y; j++) {
			OTJ_Grid_Element(initial_conditions,i,j) = sin(M_PI * i * hx);
		}
	}
	return initial_conditions;
}

void populate_analytic_solution(OTJ_Grid g, Stepsize h, int tau) {
	int i,j;
	for (i = 0; i < g.len_x; i++) {
		for (j = 0; j < g.len_y; j++) {
			OTJ_Grid_Element(g,i,j) = analytic_solution(h.x * i, h.y * j, h.t * tau);
		}
	}
}

void calculate_error(OTJ_Grid error, OTJ_Grid analytic, OTJ_Grid numerical) {
	int i,j;
	for (i = 0; i < error.len_x; i++) {
		for (j = 0; j < error.len_y; j++) {
			double a = OTJ_Grid_Element(analytic,i,j);
			double n = OTJ_Grid_Element(numerical,i,j);
			
			OTJ_Grid_Element(error,i,j) = abs(n - a);
		}
	}
}

double global_error(OTJ_Grid error) {
	double max = 0.0;
	int i,j;
	for (i = 0; i < error.len_x; i++) {
		for (j = 0; j < error.len_y; j++) {
			double e = OTJ_Grid_Element(error,i,j);
			max = (e > max) ? e : max;
		}
	}
	return max;
}

double total_error(OTJ_Grid error) {
	double total_error = 0.0;
	int i,j;
	for (i = 0; i < error.len_x; i++) {
		for (j = 0; j < error.len_y; j++) {
			total_error += OTJ_Grid_Element(error,i,j);
		}
	}
	return total_error;
}

void swap_grids(OTJ_Grid* a, OTJ_Grid* b) {
	double* temp = a->internal_storage;
	a->internal_storage = b->internal_storage;
	b->internal_storage = temp;
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
	swap_grids(&current_grid, &previous_grid);
	int tau;
	for(tau = 1; tau < go.len_t; tau++) {
		swap_grids(&current_grid, &previous_grid);
		apply_boundary_conditions(current_grid);
		solve_interior(current_grid, previous_grid,h);
	}
	OTJ_Grid solution_grid = OTJ_Grid_Alloc(go.len_x, go.len_y);
	retrieveDeviceGrid(solution_grid, current_grid);
	OTJ_Timer_Stop(tm);

	tm = OTJ_Timer_Start("Compare to analytic Solution");
	OTJ_Grid analytic_grid = OTJ_Grid_Alloc(go.len_x, go.len_y);
	populate_analytic_solution(analytic_grid,h,tau);
	// Overwrite analytic soln with error
	calculate_error(analytic_grid, analytic_grid, solution_grid);
	double gte = global_error(analytic_grid);
	double tote = total_error(analytic_grid);
	OTJ_Timer_Stop(tm);

	printf("Global Truncation Error: %f\n",gte);
	printf("Total Error: %f\n",tote);

	tm = OTJ_Timer_Start("Store Grid");
	OTJ_Grid_Store(solution_grid);
	OTJ_Timer_Stop(tm);
	
	MPI_Finalize();
	return 0;
}
