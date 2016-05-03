/* Copyright 2015 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class LSTMBlockOp : public OpKernel {
  public:
   explicit LSTMCellBlockOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
     OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
     OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));

     perftools::gputools::Stream* stream =
         ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;
      stream->
   }

   void Compute(OpKernelContext* ctx) override {
     const Tensor* x_tensor = nullptr;
     OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

     const Tensor* states_prev_tensor = nullptr;
     OP_REQUIRES_OK(ctx, ctx->input("states_prev", &states_prev_tensor));

     const Tensor* w_tensor = nullptr;
     OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

     const Tensor* b_tensor = nullptr;
     OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

     const int64 batch_size = x_tensor->dim_size(0);
     const int64 input_size = x_tensor->dim_size(1);
     const int64 state_size = cell_size_ * 7;

     perftools::gputools::Stream* stream =
         ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

     // Sanity checks for our input shapes.
     OP_REQUIRES(ctx, states_prev_tensor->dim_size(0) == batch_size,
                 errors::InvalidArgument("states_prev.dims(0) != batch_size: ",
                                         states_prev_tensor->dim_size(0),
                                         " vs. ", batch_size));
     OP_REQUIRES(ctx, states_prev_tensor->dim_size(1) == state_size,
                 errors::InvalidArgument("states_prev.dims(1) != state_size: ",
                                         states_prev_tensor->dim_size(1),
                                         " vs. ", state_size));

     OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size_,
                 errors::InvalidArgument("w.dim_size(0) != input_size + cell_size: ",
                                         w_tensor->dim_size(0),
                                         " vs. ", input_size + cell_size_));
     OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                 errors::InvalidArgument("w.dim_size(1) != cell_size * 4: ",
                                         w_tensor->dim_size(1),
                                         " vs. ", cell_size_ * 4));

     OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                 errors::InvalidArgument("b.dim_size(0) != cell_size * 4: ",
                                         b_tensor->dim_size(0),
                                         " vs. ", cell_size_ * 4));

     Tensor* h_tensor = nullptr;
     OP_REQUIRES_OK(ctx, ctx->allocate_output("h",
           TensorShape({batch_size, cell_size_}), &h_tensor));

     // Allocate our output matrices.
     Tensor* states_tensor = nullptr;
     OP_REQUIRES_OK(ctx, ctx->allocate_output("states",
         TensorShape({batch_size, state_size}), &states_tensor));

     Tensor xh_tensor;
     OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
         TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

     functor::LSTMCellBlockFprop<Device, USE_CUBLAS>()(
         stream, ctx->eigen_device<Device>(),
         batch_size, input_size, cell_size_, forget_bias_,
         x_tensor->matrix<float>(),
         xh_tensor.matrix<float>(), states_prev_tensor->matrix<float>(),
         w_tensor->matrix<float>(), b_tensor->vec<float>(),
         h_tensor->matrix<float>(), states_tensor->matrix<float>());
   }

  private:
   int64 cell_size_;
   float forget_bias_;
 };

 REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")    \
                             .Device(DEVICE_CPU),
                         LSTMCellBlockOp<CPUDevice, false>);
}

