a.out: matmul.cu
	nvcc matmul.cu
all: a.out
run: a.out
	nvprof  ./a.out
