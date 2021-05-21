#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <random>

#include "Eigen/src/Core/Matrix.h"

//#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
//#include "test_common.h"

#define TEST_DEBUG 1

using namespace Eigen;
using Matrix5d = Matrix<double, 5, 5>;


template <class C>
__host__ __device__ void printIt(C* m) {
#ifdef TEST_DEBUG
  printf("\nMatrix %dx%d\n", (int)m->rows(), (int)m->cols());
  for (u_int r = 0; r < m->rows(); ++r) {
    for (u_int c = 0; c < m->cols(); ++c) {
      printf("Matrix(%d,%d) = %f\n", r, c, (*m)(r, c));
    }
  }
#endif
}

template <class C1, class C2>
bool isEqualFuzzy(C1 a, C2 b, double epsilon = 1e-6) {
  for (unsigned int i = 0; i < a.rows(); ++i) {
    for (unsigned int j = 0; j < a.cols(); ++j) {
     if (std::abs(a(i, j) - b(i, j)) >= std::min(std::abs(a(i, j)), std::abs(b(i, j))) * epsilon) {
       printf("Failing in isEqualFuzzy: i=%d/%ld, j=%d/%ld, a(i,j)=%f, b(i,j)=%f, epsilon=%f\n", i, a.rows(), j, a.cols(), a(i,j), b(i,j), epsilon);
      }
      assert(std::abs(a(i, j) - b(i, j)) < std::min(std::abs(a(i, j)), std::abs(b(i, j))) * epsilon);
    }
  }
  return true;
}

bool isEqualFuzzy(double a, double b, double epsilon = 1e-6) {
  return std::abs(a - b) < std::min(std::abs(a), std::abs(b)) * epsilon;
}

template <typename T>
void fillMatrix(T& t) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 2.0);
  for (int row = 0; row < t.rows(); ++row) {
    for (int col = 0; col < t.cols(); ++col) {
      t(row, col) = dis(gen);
    }
  }
  return;
}

#define cudaCheck(A) assert(cudaSuccess == (A))

__global__ void kernelInverse4x4(Matrix4d *in, Matrix4d *out) { (*out) = in->inverse(); }

//__global__ void kernelInverse5x5(Matrix5d *in, Matrix5d *out) { (*out) = in->inverse(); }
__global__ void kernelInverse5x5(Matrix5d *in, Matrix5d *out) { (*out) = MatrixXd(*in).inverse(); }
//__global__ void kernelInverse5x5(Matrix5d *in, Matrix5d *out) { (*out) = Map<MatrixXd>(in->data(), 5, 5).inverse(); }

void testInverse4x4() {
  std::cout << "TEST INVERSE 4x4" << std::endl;
  Matrix4d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix4d m_inv = m.inverse();
  Matrix4d *mGPU = nullptr;
  Matrix4d *mGPUret = nullptr;
  Matrix4d *mCPUret = new Matrix4d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaCheck(cudaMalloc((void **)&mGPU, sizeof(Matrix4d)));
  cudaCheck(cudaMalloc((void **)&mGPUret, sizeof(Matrix4d)));
  cudaCheck(cudaMemcpy(mGPU, &m, sizeof(Matrix4d), cudaMemcpyHostToDevice));

  kernelInverse4x4<<<1, 1>>>(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix4d), cudaMemcpyDeviceToHost));
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
}

void testInverse5x5() {
  std::cout << "TEST INVERSE 5x5" << std::endl;
  Matrix5d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix5d m_inv = m.inverse();
  Matrix5d *mGPU = nullptr;
  Matrix5d *mGPUret = nullptr;
  Matrix5d *mCPUret = new Matrix5d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaCheck(cudaMalloc((void **)&mGPU, sizeof(Matrix5d)));
  cudaCheck(cudaMalloc((void **)&mGPUret, sizeof(Matrix5d)));
  cudaCheck(cudaMemcpy(mGPU, &m, sizeof(Matrix5d), cudaMemcpyHostToDevice));

  kernelInverse5x5<<<1, 1>>>(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaCheck(cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix5d), cudaMemcpyDeviceToHost));
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
}


int main(int argc, char *argv[]) {
  testInverse4x4();
  testInverse5x5();
  return 0;
}
