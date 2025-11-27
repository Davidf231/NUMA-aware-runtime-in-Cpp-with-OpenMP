#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>

// Función para crear matriz con un valor específico (usa constructor eficiente)
std::vector<int> createMatrix(int N, int value);

void multiplyMatrices(const std::vector<int> &A, const std::vector<int> &B, std::vector<int> &C, int N);

#endif
