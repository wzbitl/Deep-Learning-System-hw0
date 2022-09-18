#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
  size_t iteration = m / batch;
  float *X_batch = (float*)malloc(sizeof(float)*batch*n);
  unsigned char *y_batch = (unsigned char*)malloc(sizeof(unsigned char)*batch);
  float *Xthetaexp = (float*)malloc(sizeof(float)*batch*k);
  float *Xthetaexpsum = (float*)malloc(sizeof(float)*batch);
  float *I = (float*)malloc(sizeof(float)*batch*k);
  float *gradient = (float*)malloc(sizeof(float)*n*k);
  
  
  for (size_t it = 0; it < iteration; it++) {

    size_t index = it * batch;
    for (size_t i = 0; i < batch; i++) {
      y_batch[i] = y[i+index];
      for (size_t j = 0; j < n; j++) {
        X_batch[i*n + j] = X[(i + index)*n + j];
      }
    }
    for (size_t i = 0; i < batch; i++) {
      float expsum = 0;
      for (size_t j = 0; i < k; j++) {
        float sum = 0;
        for (size_t t = 0; t < n; t++) {
          sum += X_batch[i*n + t] * theta[t*k + j];
        }
        Xthetaexp[i*k + j] = exp(sum);
        expsum += Xthetaexp[i*k + j];
      }
      Xthetaexpsum[i] = expsum;
    }
    for (size_t i = 0; i < batch; i++) {
      for (size_t j = 0; j < k; j++) {
        Xthetaexp[i*k + j] /= Xthetaexpsum[i];
        I[i*k + j] = 0;
      }
    }
    for (size_t i = 0; i < batch; i++) {
      size_t one_position = y_batch[i];
      I[i*k + one_position] = 1;
    }
    for (size_t i = 0; i < batch; i++) {
      for (size_t j = 0; j < k; j++) {
        Xthetaexp[i*k + j] -= I[i*k + j];
      }
    }
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j++) {
        float sum = 0;
        for (size_t t = 0; t < batch; t++) {
          sum += X_batch[t*batch + i] * Xthetaexp[t*k + j];
        }
        gradient[i*k + j] = sum / (float)batch;
        theta[i*k + j] -= lr * gradient[i*k + j];
      }
    }
  }
  free(X_batch);
  free(y_batch);
  free(Xthetaexp);
  free(Xthetaexpsum);
  free(I);
  free(gradient);
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
