#include <stdio.h>
#include <stdlib.h>
#include "grid_options.h"

GridOptions parse_grid_options(int argc, char** argv) {
	GridOptions go;
	// Set defaults
	go.len_x = 500;
	go.len_y = 500;
	go.len_t = 10;

	int i = 1;
	while(i < argc - 1) {
		if (strcmp(argv[i],"-x") == 0) {
			go.len_x = atoi(argv[i+1]);
			i += 2;
		} else if (strcmp(argv[i],"-y") == 0) {
			go.len_y = atoi(argv[i+1]);
			i += 2;
		} else if (strcmp(argv[i],"-t") == 0) {
			go.len_t = atoi(argv[i+1]);
			i += 2;
		} else {
			printf("I'm sorry, I don't recognize the %s option.\n", argv[i]);
			printf("Options:\n");
			printf("\t-xX : Sets width of grid to X in the x direction\n");
			printf("\t-yY : Sets width of grid to Y in the y direction\n");
			printf("\t-tT : Sets number of timesteps to T\n");
			exit(1);
		}
	}
	return go;
}
