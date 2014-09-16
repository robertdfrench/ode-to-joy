include make.inc
otj.exe: main.c solve_interior.o host_grid.o device_grid.o stepsize.o grid_options.o timing_measurement.o
	$(MPICC) $(DEBUG) -o otj.exe $(CUDA_LIBRARY_PATH) -lcudart solve_interior.o device_grid.o host_grid.o stepsize.o grid_options.o timing_measurement.o main.c
	cp otj.exe $(DESTINATION)

device_grid.o: device_grid.cu grid.h
	$(NVCC) $(DEBUG) -c device_grid.cu

host_grid.o: host_grid.c grid.h
	$(MPICC) $(DEBUG) -c host_grid.c

stepsize.o: stepsize.c stepsize.h grid_options.o
	$(MPICC)  $(DEBUG) -c stepsize.c

grid_options.o: grid_options.c grid_options.h
	$(MPICC)  $(DEBUG) -c grid_options.c

solve_interior.o: solve_interior.cu solve_interior.h
	$(NVCC) $(DEBUG)  -c solve_interior.cu

timing_measurement.o: timing_measurement.c timing_measurement.h
	$(MPICC) $(DEBUG)  -c timing_measurement.c

clean:
	rm -f *.o
	rm -f *.exe
