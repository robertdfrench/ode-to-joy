include make.inc
all:
	cc -o otj.exe main.c
	cp otj.exe $(DESTINATION)
