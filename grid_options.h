#ifndef OTJ_GRID_OPTIONS
#define OTJ_GRID_OPTIONS 1
#include <stddef.h>
typedef struct grid_options_t {
	size_t len_x;
	size_t len_y;
	size_t len_t;
} GridOptions;
GridOptions parse_grid_options(int argc, char** argv);
#endif
