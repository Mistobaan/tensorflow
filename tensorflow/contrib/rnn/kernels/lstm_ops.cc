#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/rnn/kernels/lstm_ops.h"

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
#endif  // GOOGLE_CUDA


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

#endif  // GOOGLE_CUDA

void CuBlasGemm(
    OpKernelContext* ctx, perftools::gputools::Stream* stream,
    bool transa, bool transb, uint64 m, uint64 n, uint64 k, float alpha,
    const float* a, int lda, const float* b, int ldb, float beta, float *c,
    int ldc) {
#if GOOGLE_CUDA
  perftools::gputools::blas::Transpose trans[] = {
      perftools::gputools::blas::Transpose::kNoTranspose,
      perftools::gputools::blas::Transpose::kTranspose};

  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);

  bool blas_launch_status = stream->ThenBlasGemm(
      trans[transa], trans[transb], m, n, k, alpha, a_ptr, lda, b_ptr, ldb,
      beta, &c_ptr, ldc).ok();
  OP_REQUIRES(ctx, blas_launch_status, errors::Aborted("CuBlasGemm failed!"));
#else
  ctx->SetStatus(errors::InvalidArgument("CuBlasGemm needs CUDA."));
#endif
}

template <typename Device, bool USE_CUBLAS>
class LSTMCellBlockOp : public OpKernel {
 public:
  explicit LSTMCellBlockOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
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
    const int64 states_size = cell_size_ * 2;

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("states_prev.dims(0) != batch_size: ",
                                        states_prev_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(1) == states_size,
                errors::InvalidArgument("states_prev.dims(1) != cell_size * 7: ",
                                        states_prev_tensor->dim_size(1),
                                        " vs. ", states_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size_,
        errors::InvalidArgument("w.dim_size(0) != input_size + cell_size: ",
                                w_tensor->dim_size(0), " vs. ",
                                input_size + cell_size_));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                errors::InvalidArgument("w.dim_size(1) != cell_size * 4: ",
                                        w_tensor->dim_size(1),
                                        " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                errors::InvalidArgument("b.dim_size(0) != cell_size * 4: ",
                                        b_tensor->dim_size(0),
                                        " vs. ", cell_size_ * 4));

    // Allocate our output tensors.
    Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("i",
          TensorShape({batch_size, cell_size_}), &i_tensor));

    Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("cs",
          TensorShape({batch_size, cell_size_}), &cs_tensor));

    Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("f",
          TensorShape({batch_size, cell_size_}), &f_tensor));

    Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("o",
          TensorShape({batch_size, cell_size_}), &o_tensor));

    Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("ci",
          TensorShape({batch_size, cell_size_}), &ci_tensor));

    Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("co",
          TensorShape({batch_size, cell_size_}), &co_tensor));

    Tensor* states_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("states",
        TensorShape({batch_size, states_size}), &states_tensor));

    Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("h",
          TensorShape({batch_size, cell_size_}), &h_tensor));

    // Allocate our temp tensors.
    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    Tensor cs_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &cs_prev_tensor));

    Tensor h_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &h_prev_tensor));

    Tensor icfo_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_ * 4}), &icfo_tensor));

    functor::LSTMCellBlockFprop<Device, USE_CUBLAS>(
        batch_size, input_size, cell_size_)(
        ctx, stream, ctx->eigen_device<Device>(), forget_bias_,
        x_tensor->matrix<float>(), states_prev_tensor->matrix<float>(),
        w_tensor->matrix<float>(), b_tensor->vec<float>(),
        cs_prev_tensor.matrix<float>(), h_prev_tensor.matrix<float>(),
        xh_tensor.matrix<float>(),
        i_tensor->matrix<float>(), cs_tensor->matrix<float>(),
        f_tensor->matrix<float>(), o_tensor->matrix<float>(),
        ci_tensor->matrix<float>(), co_tensor->matrix<float>(),
        icfo_tensor.matrix<float>(),
        states_tensor->matrix<float>(), h_tensor->matrix<float>());
  }

 private:
  int64 cell_size_;
  float forget_bias_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")     \
                            .Device(DEVICE_CPU),  \
                        LSTMCellBlockOp<CPUDevice, false>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void TensorMemZero<GPUDevice, float>::operator()(
      const GPUDevice& d, typename TTypes<float>::Vec x);

  template <>
  void TensorMemCopy<GPUDevice, float>::operator()(
      const GPUDevice& d, typename TTypes<float>::ConstVec in,
      typename TTypes<float>::Vec out);

  template <>
  void LSTMCellBlockFprop<GPUDevice, true>::operator()(
      OpKernelContext* ctx, perftools::gputools::Stream* stream,
      const GPUDevice& d, const float forget_bias,
      typename TTypes<float>::ConstMatrix x,
      typename TTypes<float>::ConstMatrix states_prev,
      typename TTypes<float>::ConstMatrix w,
      typename TTypes<float>::ConstVec b,
      typename TTypes<float>::Matrix cs_prev,
      typename TTypes<float>::Matrix h_prev,
      typename TTypes<float>::Matrix xh,
      typename TTypes<float>::Matrix i,
      typename TTypes<float>::Matrix cs,
      typename TTypes<float>::Matrix f,
      typename TTypes<float>::Matrix o,
      typename TTypes<float>::Matrix ci,
      typename TTypes<float>::Matrix co,
      typename TTypes<float>::Matrix icfo,
      typename TTypes<float>::Matrix states,
      typename TTypes<float>::Matrix h);

  extern template struct TensorMemZero<GPUDevice, float>;
  extern template struct TensorMemCopy<GPUDevice, float>;
  extern template struct LSTMCellBlockFprop<GPUDevice, true>;
}  // end namespace functor

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")     \
                            .Device(DEVICE_GPU),  \
                        LSTMCellBlockOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA

template <typename Device, bool USE_CUBLAS>
class LSTMCellBlockGradOp : public OpKernel {
 public:
  explicit LSTMCellBlockGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bprop_dx", &bprop_dx_));
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

    const Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_tensor));

    const Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_tensor));

    const Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_tensor));

    const Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_tensor));

    const Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_tensor));

    const Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_tensor));

    const Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_tensor));

    const Tensor* h_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad_tensor));

    const Tensor* states_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states_grad", &states_grad_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 states_size = cell_size_ * 2;

    const Device& device = ctx->eigen_device<Device>();
    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("states_prev.dims(0) != batch_size: ",
                                        states_prev_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(1) == states_size,
        errors::InvalidArgument("states_prev.dims(1) != cell_size * 2: ",
                                states_prev_tensor->dim_size(1),
                                " vs. ", states_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size_,
        errors::InvalidArgument("w.dim_size(0) != input_size + cell_size: ",
                                w_tensor->dim_size(0), " vs. ",
                                input_size + cell_size_));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                errors::InvalidArgument("w.dim_size(1) != cell_size * 4: ",
                                        w_tensor->dim_size(1),
                                        " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                errors::InvalidArgument("b.dim_size(0) != cell_size * 4: ",
                                        b_tensor->dim_size(0),
                                        " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, i_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("i.dim_size(0) != batch_size: ",
                                        i_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, i_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("i.dim_size(1) != cell_size_: ",
                                        i_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, cs_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs.dim_size(0) != batch_size: ",
                                        cs_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, cs_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("cs.dim_size(1) != cell_size_: ",
                                        cs_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, f_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("f.dim_size(0) != batch_size: ",
                                        f_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, f_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("i.dim_size(1) != cell_size_: ",
                                        f_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, o_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("o.dim_size(0) != batch_size: ",
                                        o_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, o_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("o.dim_size(1) != cell_size_: ",
                                        o_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, ci_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("ci.dim_size(0) != batch_size: ",
                                        ci_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, ci_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("ci.dim_size(1) != cell_size_: ",
                                        ci_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, co_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("co.dim_size(0) != batch_size: ",
                                        co_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, co_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("co.dim_size(1) != cell_size_: ",
                                        co_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, h_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h.dim_size(0) != batch_size: ",
                                        h_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, h_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("h.dim_size(1) != cell_size_: ",
                                        h_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, h_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_grad_tensor.dims(0) != batch_size: ",
                                        h_grad_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, h_grad_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("h_grad_tensor.dims(1) != state_size: ",
                                        h_grad_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, states_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("states_grad.dims(0) != batch_size: ",
                                        states_grad_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, states_grad_tensor->dim_size(1) == states_size,
        errors::InvalidArgument("states_grad.dims(1) != cell_size * 2: ",
                                states_grad_tensor->dim_size(1),
                                " vs. ", states_size));

    // Allocate our output tensors.
    Tensor* x_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("x_grad",
        TensorShape({batch_size, input_size}), &x_grad_tensor));

    Tensor* states_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("states_prev_grad",
        TensorShape({batch_size, states_size}), &states_prev_grad_tensor));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("w_grad",
        TensorShape({input_size + cell_size_, cell_size_ * 4}),
        &w_grad_tensor));
    functor::TensorMemZero<Device, float>()(
        device, w_grad_tensor->flat<float>());

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("b_grad",
          TensorShape({cell_size_ * 4}), &b_grad_tensor));
    functor::TensorMemZero<Device, float>()(
        device, b_grad_tensor->flat<float>());

    // Allocate our temp tensors.
    Tensor cs_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &cs_prev_tensor));

    Tensor states_c_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &states_c_grad_tensor));

    Tensor states_h_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &states_h_grad_tensor));

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_grad_tensor));

    Tensor dh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &dh_tensor));

    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &di_tensor));

    Tensor dicfo_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_ * 4}), &dicfo_tensor));

    functor::LSTMCellBlockBprop<Device, USE_CUBLAS>(
        batch_size, input_size, cell_size_)(
        ctx, stream, device, bprop_dx_, x_tensor->matrix<float>(),
        states_prev_tensor->matrix<float>(), w_tensor->matrix<float>(),
        b_tensor->vec<float>(), i_tensor->matrix<float>(),
        cs_tensor->matrix<float>(), f_tensor->matrix<float>(),
        o_tensor->matrix<float>(), ci_tensor->matrix<float>(),
        co_tensor->matrix<float>(), h_tensor->matrix<float>(),
        states_grad_tensor->matrix<float>(), h_grad_tensor->matrix<float>(),
        cs_prev_tensor.matrix<float>(), states_c_grad_tensor.matrix<float>(),
        states_h_grad_tensor.matrix<float>(), xh_tensor.matrix<float>(),
        xh_grad_tensor.matrix<float>(), x_grad_tensor->matrix<float>(),
        dh_tensor.matrix<float>(), do_tensor.matrix<float>(),
        dcs_tensor.matrix<float>(), dci_tensor.matrix<float>(),
        df_tensor.matrix<float>(), di_tensor.matrix<float>(),
        dicfo_tensor.matrix<float>(), states_prev_grad_tensor->matrix<float>(),
        w_grad_tensor->matrix<float>(), b_grad_tensor->vec<float>());
  }

 protected:
  int64 cell_size_;
  bool bprop_dx_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlockGrad")    \
                            .Device(DEVICE_CPU),
                        LSTMCellBlockGradOp<CPUDevice, false>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void LSTMCellBlockBprop<GPUDevice, true>::operator()(
      OpKernelContext* ctx, perftools::gputools::Stream* stream,
      const GPUDevice& d, bool bprop_dx, typename TTypes<float>::ConstMatrix x,
      typename TTypes<float>::ConstMatrix states_prev,
      typename TTypes<float>::ConstMatrix w, typename TTypes<float>::ConstVec b,
      typename TTypes<float>::ConstMatrix i,
      typename TTypes<float>::ConstMatrix cs,
      typename TTypes<float>::ConstMatrix f,
      typename TTypes<float>::ConstMatrix o,
      typename TTypes<float>::ConstMatrix ci,
      typename TTypes<float>::ConstMatrix co,
      typename TTypes<float>::ConstMatrix h,
      typename TTypes<float>::ConstMatrix states_grad,
      typename TTypes<float>::ConstMatrix h_grad,
      typename TTypes<float>::Matrix cs_prev,
      typename TTypes<float>::Matrix states_c_grad,
      typename TTypes<float>::Matrix states_h_grad,
      typename TTypes<float>::Matrix xh, typename TTypes<float>::Matrix xh_grad,
      typename TTypes<float>::Matrix x_grad, typename TTypes<float>::Matrix dh,
      typename TTypes<float>::Matrix do_, typename TTypes<float>::Matrix dcs,
      typename TTypes<float>::Matrix dci, typename TTypes<float>::Matrix df,
      typename TTypes<float>::Matrix di, typename TTypes<float>::Matrix dicfo,
      typename TTypes<float>::Matrix states_prev_grad,
      typename TTypes<float>::Matrix w_grad,
      typename TTypes<float>::Vec b_grad);

  extern template struct LSTMCellBlockBprop<GPUDevice, true>;
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlockGrad")  \
                            .Device(DEVICE_GPU),   \
                        LSTMCellBlockGradOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA

template <typename Device, bool USE_CUBLAS>
class LSTMBlockOp : public OpKernel {
 public:
  explicit LSTMBlockOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len_tensor));

    const Tensor* initial_state_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("initial_state", &initial_state_tensor));

    OpInputList x_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("x", &x_tensors));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    OpOutputList i_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("i", &i_tensors));

    OpOutputList cs_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("cs", &cs_tensors));

    OpOutputList f_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("f", &f_tensors));

    OpOutputList o_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("o", &o_tensors));

    OpOutputList ci_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("ci", &ci_tensors));

    OpOutputList co_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("co", &co_tensors));

    OpOutputList h_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("h", &h_tensors));

    OpOutputList states_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("states", &states_tensors));

    auto sequence_len_t = sequence_len_tensor->vec<int64>();
    std::vector<int64> seq_lens_vector(sequence_len_t.size());

    const Device& device = ctx->eigen_device<Device>();
    device.memcpyDeviceToHost(
        seq_lens_vector.data(), sequence_len_t.data(),
        sizeof(int64) * sequence_len_t.size());

    const int64 batch_size = x_tensors[0].dim_size(0);
    const int64 input_size = x_tensors[0].dim_size(1);
    const int64 state_size = cell_size_ * 2;

    const int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());
    OP_REQUIRES(ctx, sequence_len_max <= sequence_len_max_,
        errors::InvalidArgument("The minibatch sequence_len_max (",
                                sequence_len_max, ") > sequence_len_max (",
                                sequence_len_max_, ")."));

    OP_REQUIRES(ctx, initial_state_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("initial_state_tensor.dims(0) == batch_size: ",
                                initial_state_tensor->dim_size(0),
                                " vs. ", batch_size));
    OP_REQUIRES(ctx, initial_state_tensor->dim_size(1) == state_size,
        errors::InvalidArgument("initial_state_tensor.dims(1) == state_size: ",
                                initial_state_tensor->dim_size(1),
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

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    for (int64 t = 0; t < sequence_len_max_; ++t ) {
      Tensor* h_tensor = nullptr;
      h_tensors.allocate(
          t, TensorShape({batch_size, cell_size_}), &h_tensor);
      functor::TensorMemZero<Device, float>()(device, h_tensor->flat<float>());

      Tensor* states_tensor = nullptr;
      states_tensors.allocate(
          t, TensorShape({batch_size, state_size}), &states_tensor);
      functor::TensorMemZero<Device, float>()(
          device, states_tensor->flat<float>());

      Tensor* i_tensor = nullptr;
      i_tensors.allocate(t, TensorShape({batch_size, cell_size_}), &i_tensor);
      Tensor* cs_tensor = nullptr;
      cs_tensors.allocate(t, TensorShape({batch_size, cell_size_}), &cs_tensor);
      Tensor* f_tensor = nullptr;
      f_tensors.allocate(t, TensorShape({batch_size, cell_size_}), &f_tensor);
      Tensor* o_tensor = nullptr;
      o_tensors.allocate(t, TensorShape({batch_size, cell_size_}), &o_tensor);
      Tensor* ci_tensor = nullptr;
      ci_tensors.allocate(t, TensorShape({batch_size, cell_size_}), &ci_tensor);
      Tensor* co_tensor = nullptr;
      co_tensors.allocate(t, TensorShape({batch_size, cell_size_}), &co_tensor);
    }

    // Allocate our temp tensors.
    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    Tensor cs_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &cs_prev_tensor));

    Tensor h_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &h_prev_tensor));

    Tensor co_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
          TensorShape({batch_size, cell_size_}), &co_tensor));

    Tensor icfo_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_ * 4}), &icfo_tensor));

    for (int64 t = 0; t < sequence_len_max; ++t) {
      const Tensor x_tensor = x_tensors[t];
      const Tensor* states_prev_tensor =
          t <= 0 ? initial_state_tensor : states_tensors[t - 1];

      Tensor* i_tensor = i_tensors[t];
      Tensor* cs_tensor = cs_tensors[t];
      Tensor* f_tensor = f_tensors[t];
      Tensor* o_tensor = o_tensors[t];
      Tensor* ci_tensor = ci_tensors[t];
      Tensor* co_tensor = co_tensors[t];
      Tensor* states_tensor = states_tensors[t];
      Tensor* h_tensor = h_tensors[t];

      functor::LSTMCellBlockFprop<Device, USE_CUBLAS>(
          batch_size, input_size, cell_size_)(
          ctx, stream, device, forget_bias_,
          x_tensor.matrix<float>(), states_prev_tensor->matrix<float>(),
          w_tensor->matrix<float>(), b_tensor->vec<float>(),
          cs_prev_tensor.matrix<float>(), h_prev_tensor.matrix<float>(),
          xh_tensor.matrix<float>(),
          i_tensor->matrix<float>(), cs_tensor->matrix<float>(),
          f_tensor->matrix<float>(), o_tensor->matrix<float>(),
          ci_tensor->matrix<float>(), co_tensor->matrix<float>(),
          icfo_tensor.matrix<float>(),
          states_tensor->matrix<float>(), h_tensor->matrix<float>());
    }
  }

 private:
  int64 sequence_len_max_;
  int64 cell_size_;
  float forget_bias_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMBlock")         \
                            .Device(DEVICE_CPU),  \
                        LSTMBlockOp<CPUDevice, false>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LSTMBlock")         \
                            .Device(DEVICE_GPU),  \
                        LSTMBlockOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA

template <typename Device, bool USE_CUBLAS>
class LSTMBlockGradOp : public OpKernel {
 public:
  explicit LSTMBlockGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bprop_dx", &bprop_dx_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len_tensor));

    const Tensor* initial_state_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("initial_state", &initial_state_tensor));

    OpInputList x_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("x", &x_tensors));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    OpInputList i_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("i", &i_tensors));

    OpInputList cs_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("cs", &cs_tensors));

    OpInputList f_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("f", &f_tensors));

    OpInputList o_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("o", &o_tensors));

    OpInputList ci_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("ci", &ci_tensors));

    OpInputList co_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("co", &co_tensors));

    OpInputList states_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("states", &states_tensors));

    OpInputList h_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("h", &h_tensors));

    OpInputList h_grad_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("h_grad", &h_grad_tensors));

    auto sequence_len_t = sequence_len_tensor->vec<int64>();
    std::vector<int64> seq_lens_vector(sequence_len_t.size());

    const Device& device = ctx->eigen_device<Device>();
    device.memcpyDeviceToHost(
        seq_lens_vector.data(), sequence_len_t.data(),
        sizeof(int64) * sequence_len_t.size());

    const int64 batch_size = x_tensors[0].dim_size(0);
    const int64 input_size = x_tensors[0].dim_size(1);
    const int64 state_size = cell_size_ * 2;

    const int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());
    OP_REQUIRES(ctx, sequence_len_max <= sequence_len_max_,
        errors::InvalidArgument("The minibatch sequence_len_max (",
                                sequence_len_max, ") > sequence_len_max (",
                                sequence_len_max_, ")."));

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

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    OpOutputList x_grad_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("x_grad", &x_grad_tensors));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("w_grad",
          TensorShape({input_size + cell_size_, cell_size_ * 4}),
          &w_grad_tensor));
    functor::TensorMemZero<Device, float>()(
        device, w_grad_tensor->flat<float>());

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("b_grad",
          TensorShape({cell_size_ * 4}), &b_grad_tensor));
    functor::TensorMemZero<Device, float>()(
        device, b_grad_tensor->flat<float>());

    for (int64 t = 0; t < sequence_len_max_; ++t) {
      Tensor* x_grad_tensor = nullptr;
      x_grad_tensors.allocate(
          t, TensorShape({batch_size, input_size}), &x_grad_tensor);
      functor::TensorMemZero<Device, float>()(
          device, x_grad_tensor->flat<float>());
    }

    // Allocate our temp tensors.
    Tensor cs_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &cs_prev_tensor));

    Tensor states_c_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &states_c_grad_tensor));

    Tensor states_h_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &states_h_grad_tensor));

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_grad_tensor));

    Tensor dh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &dh_tensor));

    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_}), &di_tensor));

    Tensor dicfo_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_ * 4}), &dicfo_tensor));

    Tensor states_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, state_size}), &states_grad_tensor));

    Tensor states_prev_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, state_size}), &states_prev_grad_tensor));
    functor::TensorMemZero<Device, float>()(
        device, states_grad_tensor.flat<float>());

    for (int64 t = sequence_len_max - 1; t >= 0; --t) {
      const Tensor& x_tensor = x_tensors[t];
      const Tensor& i_tensor = i_tensors[t];
      const Tensor& cs_tensor = cs_tensors[t];
      const Tensor& f_tensor = f_tensors[t];
      const Tensor& o_tensor = o_tensors[t];
      const Tensor& ci_tensor = ci_tensors[t];
      const Tensor& co_tensor = co_tensors[t];
      const Tensor& h_tensor = h_tensors[t];
      const Tensor& states_prev_tensor =
          t <= 0 ? *initial_state_tensor : states_tensors[t - 1];
      const Tensor& h_grad_tensor = h_grad_tensors[t];

      Tensor* x_grad_tensor = x_grad_tensors[t];
      const Tensor& states_grad_const_tensor = states_grad_tensor;

      functor::LSTMCellBlockBprop<Device, USE_CUBLAS>(
          batch_size, input_size, cell_size_)(
          ctx, stream, device, bprop_dx_, x_tensor.matrix<float>(),
          states_prev_tensor.matrix<float>(), w_tensor->matrix<float>(),
          b_tensor->vec<float>(), i_tensor.matrix<float>(),
          cs_tensor.matrix<float>(), f_tensor.matrix<float>(),
          o_tensor.matrix<float>(), ci_tensor.matrix<float>(),
          co_tensor.matrix<float>(), h_tensor.matrix<float>(),
          states_grad_const_tensor.matrix<float>(),
          h_grad_tensor.matrix<float>(), cs_prev_tensor.matrix<float>(),
          states_c_grad_tensor.matrix<float>(),
          states_h_grad_tensor.matrix<float>(), xh_tensor.matrix<float>(),
          xh_grad_tensor.matrix<float>(), x_grad_tensor->matrix<float>(),
          dh_tensor.matrix<float>(), do_tensor.matrix<float>(),
          dcs_tensor.matrix<float>(), dci_tensor.matrix<float>(),
          df_tensor.matrix<float>(), di_tensor.matrix<float>(),
          dicfo_tensor.matrix<float>(),
          states_prev_grad_tensor.matrix<float>(),
          w_grad_tensor->matrix<float>(), b_grad_tensor->vec<float>());

      const Tensor& const_states_prev_grad_tensor = states_prev_grad_tensor;
      functor::TensorMemCopy<Device, float>()(
          device, const_states_prev_grad_tensor.flat<float>(),
          states_grad_tensor.flat<float>());
    }
  }

 private:
  int64 sequence_len_max_;
  int64 cell_size_;
  bool bprop_dx_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMBlockGrad")     \
                            .Device(DEVICE_CPU),  \
                        LSTMBlockGradOp<CPUDevice, false>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LSTMBlockGrad")     \
                            .Device(DEVICE_GPU),  \
                        LSTMBlockGradOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA
}  // end namespace tensorflow
