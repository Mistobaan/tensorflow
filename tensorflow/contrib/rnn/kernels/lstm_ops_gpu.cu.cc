#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/rnn/kernels/lstm_ops.h"

namespace tensorflow {

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template struct TensorMemZero<GPUDevice, float>;
template struct TensorMemCopy<GPUDevice, float>;
template struct LSTMCellBlockFprop<GPUDevice, true>;
template struct LSTMCellBlockBprop<GPUDevice, true>;

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
