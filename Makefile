include make.inc
otj.exe: main.c solve_interior.o
	cc -o otj.exe solve_interior.o main.c
	cp otj.exe $(DESTINATION)

solve_interior.o: solve_interior.cu
	nvcc -co solve_interior.o solve_interior.cu
