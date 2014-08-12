#ifndef OTJ_GRID_HEADER
#define OTJ_GRID_HEADER 1
typedef struct OTJ_Grid_t {
	double* internal_storage;
	int len_x;
	int len_y;
} OTJ_Grid;

#define OTJ_Grid_Size(g) sizeof(double) * g.len_x * g.len_y
#define OTJ_Grid_Element(g,i,j) g.internal_storage[(i * g.len_x) + (j - 1)]

void OTJ_Grid_Store(OTJ_Grid g);
OTJ_Grid OTJ_Grid_Alloc(int len_x, int len_y);
#endif
