include make.inc
otj.exe: main.c solve_interior.o
	$(MPICC) -o otj.exe -L$(CUDA_LIBS) -lcudart solve_interior.o main.c
	cp otj.exe $(DESTINATION)

solve_interior.o: solve_interior.cu
	$(NVCC) -c solve_interior.cu

clean:
	rm -f *.o
	rm -f *.exe
