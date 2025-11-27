#include "matrixUtils.h"
#include <omp.h>
#include <iostream>
#include <vector>

const int N=12384;

int main(){
	std::vector<int> A = createMatrix(N,100);
	std::vector<int> B = createMatrix(N,15);
	std::vector<int> C = createMatrix(N,0);

	omp_set_num_threads(64);
	double start = omp_get_wtime();

	multiplyMatrices(A,B,C,N);

	double end = omp_get_wtime();

	std::cout<<end-start<<std::endl	;
}
