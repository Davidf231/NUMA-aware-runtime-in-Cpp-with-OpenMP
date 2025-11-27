#include "matrixUtils.h"
#include <omp.h>
#include <iostream>

// Constructor eficiente: crea vector con N*N elementos, todos con el mismo valor
std::vector<int> createMatrix(int N, int value) {
    return std::vector<int>(N * N, value);
}

void multiplyMatrices(const std::vector<int> &A, const std::vector<int> &B, std::vector<int> &C, int N){
	#pragma omp parallel for collapse(2)
	for(int i = 0; i < N; i++) {
	        for(int j = 0; j < N; j++) {
	                    for(int k = 0; k < N; k++) {
	                                    C[i*N+j] += A[i*N+k] * B[k*N+j];
	                                                }
	                                                        }
	                                                            }
}
