#define TILE_SIZE 4
#define KBLOCK 8
#define TILE_SIZEB 8
#define BLOCK_SIZEB 128
#define BLOCK_SIZE 128
#define KBLOCKB 8
#define STRIDE BLOCK_SIZEB/TILE_SIZEB
#define M 2560
#define N 2048
#define K 2048
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
/**
Algorithm 0: The most naive CUDA matmul kernel.
The result serves as the baseline.
*/
__global__ void matmul_0(float *A, float *B, float *C)
{
	int m_i = blockIdx.y * blockDim.y + threadIdx.y;
	int n_i = blockIdx.x * blockDim.x + threadIdx.x;
	float c = 0;
	for(int i=0; i<K; i++){
		c += A[m_i*K + i]*B[i*N + n_i];
	}
	if(m_i < M && n_i < N){
		C[m_i*N + n_i] = c;
    }
}

__global__ void matmul_1(float *A, float *B, float *C)
{
	int m_i = blockIdx.y * BLOCK_SIZE + threadIdx.y * TILE_SIZE;
	int n_i = blockIdx.x * BLOCK_SIZE + threadIdx.x * TILE_SIZE;
    float a[TILE_SIZE][KBLOCK];
    float b[KBLOCK][TILE_SIZE];
	float c[TILE_SIZE][TILE_SIZE];
    for(int i=0; i<TILE_SIZE; i++){
        for(int j=0; j<TILE_SIZE; j++){
            c[i][j]=0;
        }
    }
	for(int k_i=0; k_i<K; k_i += TILE_SIZE){
		for(int i=0; i<TILE_SIZE; i++){
            for(int k=0; k<KBLOCK; k++){
                a[i][k] = A[(m_i+i)*K + k_i + k];
                b[k][i] = B[(k_i + k)*N + n_i + i];
            }
        }
        
        for(int i=0; i<TILE_SIZE; i++){
            for(int j=0; j<TILE_SIZE; j++){
                #pragma unroll
                for(int k=0; k<KBLOCK; k++)
                    c[i][j] += a[i][k] * b[k][j];
            }
        }
	}
    for(int i=0; i<TILE_SIZE; i++){
        for(int j=0; j<TILE_SIZE; j++){
            if(m_i+i < M && n_i+j < N){
                C[(m_i+i)*N + n_i + j] = c[i][j];
            }
        }
    }

}

__global__ void matmul_2(float *A, float *B, float *C)
{
    int m_b = blockIdx.y * BLOCK_SIZE;
    int n_b = blockIdx.x * BLOCK_SIZE;
    int m_t = threadIdx.y * TILE_SIZE;
    int n_t = threadIdx.x * TILE_SIZE;
	int m_i = m_b + m_t;
	int n_i = n_b + n_t;
    float a[TILE_SIZE][KBLOCK];
    float b[KBLOCK][TILE_SIZE];
	float c[TILE_SIZE][TILE_SIZE];
    __shared__ float sa[BLOCK_SIZE*KBLOCK];
    __shared__ float sb[KBLOCK*BLOCK_SIZE];
    
    for(int i=0; i<TILE_SIZE; i++){
        for(int j=0; j<TILE_SIZE; j++){
            c[i][j]=0;
        }
    }
    
    
	for(int k_i=0; k_i<K; k_i += KBLOCK){
        
        for (int t=threadIdx.y*blockDim.x + threadIdx.x; t<BLOCK_SIZE*KBLOCK; t += blockDim.x*blockDim.y){
            sa[t] = A[(m_b + t/KBLOCK)*K + k_i + t%KBLOCK];
            sb[t] = B[(k_i + t/BLOCK_SIZE)*N + n_b + t%BLOCK_SIZE];
        }
        __syncthreads();
		for(int i=0; i<TILE_SIZE; i++){
            for(int k=0; k<KBLOCK; k++){
                a[i][k] = sa[(m_t + i)*KBLOCK + k];
                b[k][i] = sb[k*BLOCK_SIZE + n_t + i];
            }
        }
        
        for(int i=0; i<TILE_SIZE; i++){
            for(int j=0; j<TILE_SIZE; j++){
                #pragma unroll
                for(int k=0; k<KBLOCK; k++)
                    c[i][j] += a[i][k] * b[k][j];
            }
        }
        __syncthreads();
	}
    for(int i=0; i<TILE_SIZE; i++){
        for(int j=0; j<TILE_SIZE; j++){
            if(m_i+i < M && n_i+j < N){
                C[(m_i+i)*N + n_i + j] = c[i][j];
            }
        }
    }

}

__global__ void matmul_3(float *A, float *B, float *C)
{
    int m_b = blockIdx.y * BLOCK_SIZEB;
    int n_b = blockIdx.x * BLOCK_SIZEB;
    int m_t = threadIdx.y * TILE_SIZEB;
    int n_t = threadIdx.x * TILE_SIZEB;
	int m_i = m_b + m_t;
	int n_i = n_b + n_t;
    float a[TILE_SIZEB];
    float b[TILE_SIZEB];
	float c[TILE_SIZEB][TILE_SIZEB];
    __shared__ float sa[BLOCK_SIZEB*KBLOCKB];
    __shared__ float sb[KBLOCKB*BLOCK_SIZEB];
    
    for(int i=0; i<TILE_SIZEB; i++){
        for(int j=0; j<TILE_SIZEB; j++){
            c[i][j]=0;
        }
    }
    
    
	for(int k_i=0; k_i<K; k_i += KBLOCKB){
        
        for (int t=threadIdx.y*blockDim.x + threadIdx.x; t<BLOCK_SIZEB*KBLOCKB; t += blockDim.x*blockDim.y){
            sa[t] = A[(m_b + t/KBLOCKB)*K + k_i + t%KBLOCKB];
            sb[t] = B[(k_i + t/BLOCK_SIZEB)*N + n_b + t%BLOCK_SIZEB];
        }
        __syncthreads();
        for(int k=0; k<KBLOCKB; k++){
            #pragma unroll
            for(int i=0; i<TILE_SIZEB; i++){
                a[i] = sa[(m_t + i)*KBLOCKB + k];
                b[i] = sb[k*BLOCK_SIZEB + n_t + i];
            }
            
            for(int i=0; i<TILE_SIZEB; i++){
                #pragma unroll
                for(int j=0; j<TILE_SIZEB; j++){
                    c[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
	}
    for(int i=0; i<TILE_SIZEB; i++){
        for(int j=0; j<TILE_SIZEB; j++){
            if(m_i+i < M && n_i+j < N){
                C[(m_i+i)*N + n_i + j] = c[i][j];
            }
        }
    }

}

__global__ void matmul_4(float *A, float *B, float *C)
{
    int m_b = blockIdx.y * BLOCK_SIZEB;
    int n_b = blockIdx.x * BLOCK_SIZEB;

	int m_i = m_b + threadIdx.y;
	int n_i = n_b + threadIdx.x;
    float a[TILE_SIZEB];
    float b[TILE_SIZEB];
	float c[TILE_SIZEB][TILE_SIZEB];
    __shared__ float sa[BLOCK_SIZEB*KBLOCKB];
    __shared__ float sb[KBLOCKB*BLOCK_SIZEB];
    
    
    for(int i=0; i<TILE_SIZEB; i++){
        for(int j=0; j<TILE_SIZEB; j++){
            c[i][j]=0;
        }
    }
    
    
	for(int k_i=0; k_i<K; k_i += KBLOCKB){
        
        for (int t=threadIdx.y*STRIDE + threadIdx.x; t<BLOCK_SIZEB*KBLOCKB; t += STRIDE*STRIDE){
            sa[t] = A[(m_b + t/KBLOCKB)*K + k_i + t%KBLOCKB];
            sb[t] = B[(k_i + t/BLOCK_SIZEB)*N + n_b + t%BLOCK_SIZEB];
        }
        __syncthreads();
        for(int k=0; k<KBLOCKB; k++){
            #pragma unroll
            for(int i=0; i<TILE_SIZEB; i++){
                a[i] = sa[(i*STRIDE + threadIdx.y)*KBLOCKB + k];
                b[i] = sb[k*BLOCK_SIZEB + i*STRIDE + threadIdx.x];
            }
            
            for(int i=0; i<TILE_SIZEB; i++){
                #pragma unroll
                for(int j=0; j<TILE_SIZEB; j++){
                    c[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
	}
    for(int i=0; i<TILE_SIZEB; i++){
        for(int j=0; j<TILE_SIZEB; j++){
            if(m_i+i*STRIDE < M && n_i+j*STRIDE < N){
                C[(m_i+i*STRIDE)*N + n_i + j*STRIDE] = c[i][j];
            }
        }
    }

}

__global__ void matmul_5(float *A, float *B, float *C)
{
    int m_b = blockIdx.y * BLOCK_SIZEB;
    int n_b = blockIdx.x * BLOCK_SIZEB;

	int m_i = m_b + threadIdx.y;
	int n_i = n_b + threadIdx.x;
    float a[TILE_SIZEB][KBLOCKB];
    float b[KBLOCKB][TILE_SIZEB];
	float c[TILE_SIZEB][TILE_SIZEB];
    __shared__ float sa[BLOCK_SIZEB*KBLOCKB];
    __shared__ float sb[KBLOCKB*BLOCK_SIZEB];
    
    for(int i=0; i<TILE_SIZEB; i++){
        for(int j=0; j<TILE_SIZEB; j++){
            c[i][j]=0;
        }
    }
    
    
	for(int k_i=0; k_i<K; k_i += KBLOCKB){
        
        for (int t=threadIdx.y*STRIDE + threadIdx.x; t<BLOCK_SIZEB*KBLOCKB; t += STRIDE*STRIDE){
            sa[t] = A[(m_b + t/KBLOCKB)*K + k_i + t%KBLOCKB];
            sb[t] = B[(k_i + t/BLOCK_SIZEB)*N + n_b + t%BLOCK_SIZEB];
        }
        __syncthreads();
		for(int i=0; i<TILE_SIZEB; i++){
            for(int k=0; k<KBLOCKB; k++){
                a[i][k] = sa[(i*STRIDE + threadIdx.y)*KBLOCKB + k];
                b[k][i] = sb[k*BLOCK_SIZEB + i*STRIDE + threadIdx.x];
            }
        }
        
        for(int i=0; i<TILE_SIZEB; i++){
            for(int j=0; j<TILE_SIZEB; j++){
                #pragma unroll
                for(int k=0; k<KBLOCKB; k++)
                    c[i][j] += a[i][k] * b[k][j];
            }
        }
        __syncthreads();
	}
    for(int i=0; i<TILE_SIZEB; i++){
        for(int j=0; j<TILE_SIZEB; j++){
            if(m_i+i*STRIDE < M && n_i+j*STRIDE < N){
                C[(m_i+i*STRIDE)*N + n_i + j*STRIDE] = c[i][j];
            }
        }
    }

}

int main(int argc, char *argv[])
{
	srand(0);
	float *cpu_a = (float*)malloc(sizeof(float)*M*K);
	float *cpu_b = (float*)malloc(sizeof(float)*K*N);
	float *cpu_c = (float*)malloc(sizeof(float)*M*N);
	float *cpu_c2 = (float*)malloc(sizeof(float)*M*N);

	for (int i=0; i<M*K; i++)
		cpu_a[i] = rand()/65536;

	for (int i=0; i<N*K; i++)
		cpu_b[i] = rand()/65536;

	//cudaEvent_t start, end;
	//cudaEventCreate(&start);
	//cudeEventCreate(&end);

	float *gpu_a, *gpu_b, *gpu_c;
	cudaMalloc((void**)&gpu_a, sizeof(float)*M*K);
	cudaMalloc((void**)&gpu_b, sizeof(float)*N*K);
	cudaMalloc((void**)&gpu_c, sizeof(float)*M*N);
	
	cudaMemcpy(gpu_a, cpu_a, sizeof(float)*M*K, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, cpu_b, sizeof(float)*N*K, cudaMemcpyHostToDevice);

	dim3 grid0(M/TILE_SIZE,N/TILE_SIZE);
    dim3 block0(TILE_SIZE, TILE_SIZE);

	matmul_0<<<grid0, block0>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();
    printf("%s\n",cudaGetErrorString(cudaPeekAtLastError()));
	cudaMemcpy(cpu_c2, gpu_c, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	//cudaEventRecord(start, 0);
    dim3 grid(M/BLOCK_SIZE,N/BLOCK_SIZE);
    dim3 block(BLOCK_SIZE/TILE_SIZE, BLOCK_SIZE/TILE_SIZE);

	matmul_2<<<grid, block>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();
    printf("%s\n",cudaGetErrorString(cudaPeekAtLastError()));
	cudaMemcpy(cpu_c, gpu_c, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    matmul_1<<<grid, block>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();
    printf("%s\n",cudaGetErrorString(cudaPeekAtLastError()));
	cudaMemcpy(cpu_c, gpu_c, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	
    dim3 gridb(M/BLOCK_SIZEB,N/BLOCK_SIZEB);
    dim3 blockb(BLOCK_SIZEB/TILE_SIZEB, BLOCK_SIZEB/TILE_SIZEB);

    matmul_4<<<gridb, blockb>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();
    printf("%s\n",cudaGetErrorString(cudaPeekAtLastError()));
	cudaMemcpy(cpu_c, gpu_c, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
	matmul_5<<<gridb, blockb>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();
    printf("%s\n",cudaGetErrorString(cudaPeekAtLastError()));
	cudaMemcpy(cpu_c, gpu_c, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    float sum = 0.0;
    for (int i=0; i<M*N; i++)
		sum += cpu_c2[i];
    printf("%f\n",sum);
    sum = 0.0;
    for (int i=0; i<M*N; i++)
		sum += cpu_c[i];
    printf("%f\n",sum);
    sum = 0.0;
    for (int i=0; i<M*N; i++)
		sum += cpu_c[i] - cpu_c2[i];
    printf("%f\n",sum);
	//cudaEventRecord(end,0);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
	free(cpu_a);
	free(cpu_b);
	free(cpu_c);
    free(cpu_c2);
}
