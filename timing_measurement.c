#include <mpi.h>
#include <stdio.h>
#include "timing_measurement.h"

void OTJ_Timer_Stop(OTJ_Timer tm) {
	printf("[Timer] %s: %f seconds\n", tm.message, MPI_Wtime() - tm.beginning);
}

OTJ_Timer OTJ_Timer_Start(char* message) {
	OTJ_Timer tm;
	tm.message = message;
	tm.beginning = MPI_Wtime();
	tm.end = tm.beginning;
	return tm;
}
