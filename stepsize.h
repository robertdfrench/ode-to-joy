#ifndef OTJ_STEPSIZE_HEADER
#define OTJ_STEPSIZE_HEADER 1
#include "grid_options.h"
typedef struct stepsize_t {
	float x;
	float y;
	float t;
} Stepsize;
Stepsize stepsize_from_grid_options(GridOptions go);
#endif
