#ifndef OTJ_SOLVE_INTERIOR
#define OTJ_SOLVE_INTERIOR 1
void solve_interior(OTJ_Grid current, OTJ_Grid previous, Stepsize h);
OTJ_Grid createDeviceGrid(OTJ_Grid host_grid);
OTJ_Grid createAndCopyDeviceGrid(OTJ_Grid host_grid);
void retrieveDeviceGrid(OTJ_Grid host_grid, OTJ_Grid device_grid);
void solve_interior(OTJ_Grid current, OTJ_Grid previous, Stepsize h);
void apply_boundary_conditions(OTJ_Grid g);
#endif
