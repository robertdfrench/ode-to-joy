include make.inc
otj.exe: main.c solve_interior.o grid.o stepsize.o grid_options.o timing_measurement.o
	$(MPICC) $(DEBUG) -o otj.exe $(CUDA_LIBRARY_PATH) -lcudart solve_interior.o grid.o stepsize.o grid_options.o timing_measurement.o main.c
	cp otj.exe $(DESTINATION)

grid.o: grid.c grid.h
	$(MPICC)  $(DEBUG) -c grid.c

stepsize.o: stepsize.c stepsize.h grid_options.o
	$(MPICC)  $(DEBUG) -c stepsize.c

grid_options.o: grid_options.c grid_options.h
	$(MPICC)  $(DEBUG) -c grid_options.c

solve_interior.o: solve_interior.cu
	$(NVCC) $(DEBUG)  -c solve_interior.cu

timing_measurement.o: timing_measurement.c timing_measurement.h
	$(MPICC) $(DEBUG)  -c timing_measurement.c

clean:
	rm -f *.o
	rm -f *.exe
