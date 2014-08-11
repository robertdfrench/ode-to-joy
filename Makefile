include make.inc
otj.exe: main.c solve_interior.o grid.o stepsize.o grid_options.o
	$(MPICC) -o otj.exe $(CUDA_LIBRARY_PATH) -lcudart solve_interior.o grid.o stepsize.o grid_options.o main.c
	cp otj.exe $(DESTINATION)

grid.o: grid.c grid.h
	$(MPICC) -c grid.c

stepsize.o: stepsize.c stepsize.h grid_options.o
	$(MPICC) -c stepsize.c

grid_options.o: grid_options.c grid_options.h
	$(MPICC) -c grid_options.c

solve_interior.o: solve_interior.cu
	$(NVCC) -c solve_interior.cu

clean:
	rm -f *.o
	rm -f *.exe
