#ifndef OTJ_SOLVE_INTERIOR
#define OTJ_SOLVE_INTERIOR 1
void solve_interior(Grid current, Grid previous, Stepsize h);
Grid createDeviceGrid(Grid host_grid);
Grid createAndCopyDeviceGrid(Grid host_grid);
void retrieveDeviceGrid(Grid host_grid, Grid device_grid);
void solve_interior(Grid current, Grid previous, Stepsize h);
void apply_boundary_conditions(Grid g);
#endif
