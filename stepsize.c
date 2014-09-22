#include "stepsize.h"

Stepsize stepsize_from_grid_options(GridOptions go) {
	Stepsize h;
	// Add one to the denominator because both
	// spatial boundaries are part of the problem
	h.x = 1.0 / ((double) go.len_x + 1);
	h.y = 1.0 / ((double) go.len_y + 1);
	h.t = 1.0 / ((double) go.len_t);
	return h;
}
