#include "stepsize.h"

Stepsize stepsize_from_grid_options(GridOptions go) {
	Stepsize h;
	h.x = 1.0 / ((double) go.len_x);
	h.y = 1.0 / ((double) go.len_y);
	h.t = 1.0 / ((double) go.len_t);
	return h;
}
