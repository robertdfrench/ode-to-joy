include make.inc
otj.exe: main.c solve_interior.o grid.o stepsize.o
	$(MPICC) -o otj.exe $(CUDA_LIBRARY_PATH) -lcudart solve_interior.o grid.o stepsize.o main.c
	cp otj.exe $(DESTINATION)

grid.o: grid.c grid.h
	$(MPICC) -c grid.c

stepsize.o: stepsize.c stepsize.h
	$(MPICC) -c stepsize.c

solve_interior.o: solve_interior.cu
	$(NVCC) -c solve_interior.cu

clean:
	rm -f *.o
	rm -f *.exe
