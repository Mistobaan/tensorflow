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
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

int64 GetCudnnWorkspaceLimit(const string& envvar_in_mb,
                             int64 default_value_in_bytes);

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchRNNOp;

template <typename T>
struct LaunchRNNOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, bool use_cudnn,
    const Tensor& x, const Tensor& hx, const Tensor& cx,
    Tensor* y, Tensor* hy, Tensor *cy, Tensor *dropout_state) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    if (!use_cudnn) {
      // TODO(fmilo) use cublas impl
      return;
    }

     perftools::gputools::dnn::RNNDescriptor rnn_descriptor;

     // input
     // Tensor* x_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("x",
     //       TensorShape({batch_size, cell_size_}), &x_tensor));

     // Tensor* hx_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("hx",
     //       TensorShape({batch_size, cell_size_}), &hx_tensor));

     // Tensor* cx_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("cx",
     //       TensorShape({batch_size, cell_size_}), &cx_tensor));

     // Tensor* w_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("w",
     //       TensorShape({batch_size, cell_size_}), &w_tensor));

     // // output
     // Tensor* y_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("y",
     //       TensorShape({batch_size, cell_size_}), &cx_tensor));

     // Tensor* hy_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("hy",
     //       TensorShape({batch_size, cell_size_}), &hy_tensor));

     // Tensor* cy_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("cy",
     //       TensorShape({batch_size, cell_size_}), &cy_tensor));

     // Tensor* output_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("output_tensor",
     //       TensorShape({batch_size, cell_size_}), &output_tensor));

     // Dropout
     // Tensor* dropout_state_tensor = nullptr;
     // OP_REQUIRES_OK(ctx, ctx->allocate_output("dropout_state",
     //       TensorShape({batch_size, cell_size_}), &dropout_state_tensor));

      auto x_ptr = AsDeviceMemory(x.template flat<T>().data(),
                                  x.template flat<T>().size());
      auto hx_ptr = AsDeviceMemory(hx.template flat<T>().data(),
                                  hx.template flat<T>().size());
      auto cx_ptr = AsDeviceMemory(hx.template flat<T>().data(),
                                  hx.template flat<T>().size());

      auto y_ptr = AsDeviceMemory(y->template flat<T>().data(),
                                  y->template flat<T>().size());
      auto hy_ptr = AsDeviceMemory(hy->template flat<T>().data(),
                                   hy->template flat<T>().size());
      auto cy_ptr = AsDeviceMemory(cy->template flat<T>().data(),
                                   cy->template flat<T>().size());

      auto dropout_state_ptr = AsDeviceMemory(dropout_state->template flat<T>().data(),
                                              dropout_state->template flat<T>().size());

    static int64 ConvolveScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB by default
        );

    CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);

     bool rnn_launch_status = stream->ThenRNNFoward(
        rnn_descriptor,
        x_ptr,
        hx_ptr,
        w_ptr,
        y_ptr,
        hy_ptr,
        cy_ptr,
        dropout_state_ptr,
        scratch_allocator
      ).ok();

     if (!rnn_launch_status) {
        ctx->SetStatus(errors::Internal("cudnnRNNForward launch failure :"
            " x shape(", x.shape().DebugString(),
            ") hx shape(", hx.shape().DebugString(),
            ") cx shape(", cx.shape().DebugString(),
            ") y shape(", y->shape().DebugString(),
            ") hy shape(", hy->shape().DebugString(),
            ") cy shape(", cy->shape().DebugString(), ")"
            ));
        return;
      };

   }
 };
