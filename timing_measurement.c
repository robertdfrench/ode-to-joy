#include <mpi.h>
#include <stdio.h>
#include "timing_measurement.h"

void stop_timer(TimingMeasurement tm) {
	printf("[Timer] %s: %f seconds\n", tm.message, MPI_Wtime() - tm.beginning);
}

TimingMeasurement start_timer(char* message) {
	TimingMeasurement tm;
	tm.message = message;
	tm.beginning = MPI_Wtime();
	tm.end = tm.beginning;
	return tm;
}
