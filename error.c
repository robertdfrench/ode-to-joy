#include "grid.h"
#include "error.h"
#include "stdlib.h"

void OTJ_Calculate_Error(OTJ_Grid error, OTJ_Grid analytic, OTJ_Grid numerical) {
	int i,j;
	for (i = 0; i < error.len_x; i++) {
		for (j = 0; j < error.len_y; j++) {
			double a = OTJ_Grid_Element(analytic,i,j);
			double n = OTJ_Grid_Element(numerical,i,j);
			
			OTJ_Grid_Element(error,i,j) = abs(n - a);
		}
	}
}

double OTJ_Global_Error(OTJ_Grid error) {
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

double OTJ_Total_Error(OTJ_Grid error) {
	double total_error = 0.0;
	int i,j;
	for (i = 0; i < error.len_x; i++) {
		for (j = 0; j < error.len_y; j++) {
			total_error += OTJ_Grid_Element(error,i,j);
		}
	}
	return total_error;
}
