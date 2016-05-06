#ifndef TENSORFLOW_KERNELS_LSTM_OPS_H_
#define TENSORFLOW_KERNELS_LSTM_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace perftools {
namespace gputools {
class Stream;
}  // end namespace gputools
}  // end namespace perftools

namespace tensorflow {
class OpKernelContext;

void CuBlasGemm(
    OpKernelContext* ctx, perftools::gputools::Stream* stream, bool transa,
    bool transb, uint64 m, uint64 n, uint64 k, float alpha, const float* a,
    int lda, const float* b, int ldb, float beta, float* c, int ldc);

namespace functor {

template <typename Device, typename T>
struct TensorMemZero {
  void operator()(const Device& d, typename TTypes<T>::Vec x) {
    x.device(d) = x.constant(0);
  }
};

template <typename Device, typename T>
struct TensorMemCopy {
  void operator()(const Device& d, typename TTypes<T>::ConstVec in,
                  typename TTypes<T>::Vec out) {
    out.device(d) = in;
  }
};

template <typename Device, bool USE_CUBLAS>
struct TensorBlasGemm {
  template <typename A_T, typename B_T>
  void operator()(
      OpKernelContext* ctx, perftools::gputools::Stream* stream,
      const Device& d, bool transa, bool transb, float alpha,
      A_T a, B_T b, float beta, typename TTypes<float>::Matrix c) {
    if (USE_CUBLAS) {
      int64 m = c.dimensions()[0];
      int64 n = c.dimensions()[1];
      int64 k = transa ? a.dimensions()[0] : a.dimensions()[1];

      CuBlasGemm(ctx, stream, transb, transa, n, m, k, alpha, b.data(),
                 transb ? k : n, a.data(), transa ? m : k, beta, c.data(), n);
    } else {
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
      contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(
          transa == false, transb == true);
      if (alpha == 1.0f && beta == 0.0f) {
        c.device(d) = a.contract(b, contract_pairs);
      } else if (alpha == 1.0f && beta == 1.0f) {
        c.device(d) += a.contract(b, contract_pairs);
      } else {
        c.device(d) = c.constant(alpha) * a.contract(b, contract_pairs) +
                      c.constant(beta) * c;
      }
    }
  }
};

struct LSTMCellBlock {
  LSTMCellBlock(const int batch_size, const int input_size, const int cell_size)
    : batch_size_(batch_size), input_size_(input_size), cell_size_(cell_size) {}

  Eigen::array<int, 2> icfo_i_offsets() const {
    return {0, cell_size_ * 0};
  }

  Eigen::array<int, 2> icfo_c_offsets() const {
    return {0, cell_size_ * 1};
  }

  Eigen::array<int, 2> icfo_f_offsets() const {
    return {0, cell_size_ * 2};
  }

  Eigen::array<int, 2> icfo_o_offsets() const {
    return {0, cell_size_ * 3};
  }

  Eigen::array<int, 2> states_cs_offsets() const {
    return {0, 0};
  }

  Eigen::array<int, 2> states_h_offsets() const {
    return {0, cell_size_};
  }

  Eigen::array<int, 2> cell_extents() const {
    return {batch_size_, cell_size_};
  }

  Eigen::array<int, 2> xh_x_offsets() const {
    return {0, 0};
  }

  Eigen::array<int, 2> xh_x_extents() const {
    return {batch_size_, input_size_};
  }

  Eigen::array<int, 2> xh_h_offsets() const {
    return {0, input_size_};
  }

  Eigen::array<int, 2> xh_h_extents() const {
    return {batch_size_, cell_size_};
  }

 protected:
  const int batch_size_;
  const int input_size_;
  const int cell_size_;
};

template <typename Device, bool USE_CUBLAS>
struct LSTMCellBlockFprop : public LSTMCellBlock {
  LSTMCellBlockFprop(const int batch_size, const int input_size,
                     const int cell_size)
    : LSTMCellBlock(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, perftools::gputools::Stream* stream,
      const Device& d, const float forget_bias,
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
      typename TTypes<float>::Matrix h) {
    // [cs, h] = states_prev
    cs_prev.device(d) =
        states_prev.slice(states_cs_offsets(), cell_extents());

    h_prev.device(d) =
        states_prev.slice(states_h_offsets(), cell_extents());

    // Concat xh = [x, h].
    xh.slice(xh_x_offsets(), xh_x_extents()).device(d) = x;
    xh.slice(xh_h_offsets(), xh_h_extents()).device(d) = h_prev;

    // states1 = xh * w + b
    TensorBlasGemm<Device, USE_CUBLAS>()(
        ctx, stream, d, false, false, 1.0f, xh, w, 0.0f, icfo);
    icfo.device(d) +=
        b.broadcast(Eigen::array<int, 2>({batch_size_, 1}));

    // Input gate.
    i.device(d) = icfo.slice(icfo_i_offsets(), cell_extents()).sigmoid();

    // Cell input.
    ci.device(d) = icfo.slice(icfo_c_offsets(), cell_extents()).tanh();

    // Forget gate (w/ bias).
    f.device(d) =
        (icfo.slice(icfo_f_offsets(), cell_extents()) + f.constant(forget_bias))
        .sigmoid();

    // cs = ci .* i + f .* cs_prev
    cs.device(d) = i * ci + f * cs_prev;

    // co = tanh(cs)
    co.device(d) = cs.tanh();

    // Output gate.
    o.device(d) = icfo.slice(icfo_o_offsets(), cell_extents()).sigmoid();

    // h = o .* co
    h.device(d) = o * co;

    states.slice(states_cs_offsets(), cell_extents()).device(d) = cs;
    states.slice(states_h_offsets(), cell_extents()).device(d) = h;
  }
};

template <typename Device, bool USE_CUBLAS>
struct LSTMCellBlockBprop : public LSTMCellBlock {
  LSTMCellBlockBprop(const int batch_size, const int input_size,
                     const int cell_size)
    : LSTMCellBlock(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, perftools::gputools::Stream* stream,
      const Device& d, bool bprop_dx, typename TTypes<float>::ConstMatrix x,
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
      typename TTypes<float>::Vec b_grad) {
    // [c_grad, h_grad] = states_grad.
    states_c_grad.device(d) =
        states_grad.slice(states_cs_offsets(), cell_extents());
    states_h_grad.device(d) =
        states_grad.slice(states_h_offsets(), cell_extents());

    // dh.
    dh.device(d) = h_grad + states_h_grad;

    // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
    do_.device(d) = o * (o.constant(1.0f) - o) * dh * co;

    // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
    dcs.device(d) =
          (co.constant(1.0f) - co * co) * dh * o + states_c_grad;

    // dci[t] = tanh'(ci[t]) dcs[t] i[t]
    dci.device(d) = (ci.constant(1.0f) - ci * ci) * dcs * i;

    // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
    cs_prev.device(d) = states_prev.slice(states_cs_offsets(), cell_extents());
    df.device(d) = f * (f.constant(1.0f) - f) * dcs * cs_prev;

    // di[t] = sigm'(i[t]) dcs[t] ci[t]
    di.device(d) = i * (i.constant(1.0f) - i) * dcs * ci;

    dicfo.slice(icfo_i_offsets(), cell_extents()).device(d) = di;
    dicfo.slice(icfo_c_offsets(), cell_extents()).device(d) = dci;
    dicfo.slice(icfo_f_offsets(), cell_extents()).device(d) = df;
    dicfo.slice(icfo_o_offsets(), cell_extents()).device(d) = do_;

    // We can parallelize the bprop GEMMs on the CPU (on GPU doesn't make any
    // difference).
    BlockingCounter counter(2);
    auto workers_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    auto workers = workers_threads.workers;

    // xh_grad = dstate4 * w^T
    workers->Schedule([ctx, stream, d, dicfo, w, xh_grad, &counter]() {
      TensorBlasGemm<Device, USE_CUBLAS>()(
          ctx, stream, d, false, true, 1.0f, dicfo, w, 0.0f, xh_grad);
      counter.DecrementCount();
    });

    // Concat xh = [x, h].
    auto h_prev = states_prev.slice(states_h_offsets(), cell_extents());
    xh.slice(xh_x_offsets(), xh_x_extents()).device(d) = x;
    xh.slice(xh_h_offsets(), xh_h_extents()).device(d) = h_prev;

    // w_grad, b_grad.
    workers->Schedule([ctx, stream, d, xh, dicfo, w_grad, &counter]() {
      TensorBlasGemm<Device, USE_CUBLAS>()(
          ctx, stream, d, true, false, 1.0f, xh, dicfo, 1.0f, w_grad);
      counter.DecrementCount();
    });
    b_grad.device(d) += dicfo.sum(Eigen::array<int, 1>({0}));

    // Need to make sure our GEMMs are done.
    counter.Wait();

    // x_grad.
    x_grad.device(d) = xh_grad.slice(xh_x_offsets(), xh_x_extents());

    // states_prev_grad = [dcs, dh]
    states_prev_grad.slice(states_cs_offsets(), cell_extents()).device(d) = dcs * f;
    states_prev_grad.slice(states_h_offsets(), cell_extents()).device(d) =
        xh_grad.slice(xh_h_offsets(), xh_h_extents());
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_LSTM_OPS_H_
