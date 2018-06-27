a.out: matmul.cu
	nvcc matmul.cu
all: a.out
run: a.out
	nvprof --print-gpu-trace ./a.out
