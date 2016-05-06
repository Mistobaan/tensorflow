#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("LSTMCellBlock")
    .Attr("cell_size: int")
    .Attr("forget_bias: float = 1.0")
    .Attr("bprop_dx: bool = true")
    .Input("x: float")
    .Input("states_prev: float")
    .Input("w: float")
    .Input("b: float")
    .Output("i: float")
    .Output("cs: float")
    .Output("f: float")
    .Output("o: float")
    .Output("ci: float")
    .Output("co: float")
    .Output("states: float")
    .Output("h: float")
    .Doc(R"doc(
Computes the LSTM cell forward propagation for 1 time step.

This implementation uses 1 weight matrix and 1 bias vector, there is no
diagonal peephole connection.

This kernel op implements the following mathematical equations:

  [_, cs_prev, _, _, _, _, h_prev] = states

  xh = [x, h_prev]
  [i, f, ci, o] = xh * w + b
  f = f + forget_bias

  i = sigm(i)
  f = sigm(f)
  ci = tanh(ci)
  o = sigm(o)

  cs = ci .* i + cs_prev .* f
  co = tanh(cs)

  h = co .* o
  states = [i, cs, f, o, ci, co, h]


cell_size: The LSTM cell size.
forget_bias: The forget gate bias.
x: The input to the LSTM cell.
states_prev: The previous LSTM state of i, cs, f, o, ci, co, h.
w: The weight matrix.
b: The bias vector.
states: The LSTM state of i, cs, f, o, ci, co, h.
h: The output h vector.
)doc");

REGISTER_OP("LSTMCellBlockGrad")
    .Attr("cell_size: int")
    .Attr("bprop_dx: bool")
    .Input("x: float")
    .Input("states_prev: float")
    .Input("w: float")
    .Input("b: float")
    .Input("i: float")
    .Input("cs: float")
    .Input("f: float")
    .Input("o: float")
    .Input("ci: float")
    .Input("co: float")
    .Input("h: float")
    .Input("states_grad: float")
    .Input("h_grad: float")
    .Output("x_grad: float")
    .Output("states_prev_grad: float")
    .Output("w_grad: float")
    .Output("b_grad: float")
    .Doc(R"doc(
Computes the LSTM cell backward propagation for 1 timestep.

This implementation is to be used inconjunction of LSTMCellBlock.

cell_size: The LSTM cell size.
x: The input to the LSTM cell.
states_prev: The previous LSTM state (it is a concatenated vector of c[t - 1]
  and h[t - 1].
w: The weight matrix.
b: The bias vector.
h: The output h[t] vector.
states: The state vector (it is the concatenated vector of c[t] and h[t].
states_grad: The gradient of states vector.
x_grad: The gradient of x.
states_prev_grad: The gradient of states_prev.
w_grad: The gradient of w.
b_grad: The gradient of b.
)doc");

REGISTER_OP("LSTMBlock")
    .Attr("cell_size: int")
    .Attr("forget_bias: float = 1.0")
    .Attr("sequence_len_max: int")
    .Attr("bprop_dx: bool = true")
    .Input("sequence_len: int64")
    .Input("initial_state: float")
    .Input("x: sequence_len_max * float")
    .Input("w: float")
    .Input("b: float")
    .Output("i: sequence_len_max * float")
    .Output("cs: sequence_len_max * float")
    .Output("f: sequence_len_max * float")
    .Output("o: sequence_len_max * float")
    .Output("ci: sequence_len_max * float")
    .Output("co: sequence_len_max * float")
    .Output("states: sequence_len_max * float")
    .Output("h: sequence_len_max * float")
    .Doc(R"doc(
Computes the LSTM forward propagation for N time steps.

This implementation uses 1 weight matrix and 1 bias vector, there is no
diagonal peephole connection. The computation of this op is dynamic as a
function of sequence_len. We compute N = max(sequence_len) timesteps.

cell_size: The LSTM cell size.
forget_bias: The forget gate bias.
sequence_len: A vector of batch_size containing the sequence length.
initial_state: Initial state of the LSTM.
x: The list of inputs to the LSTM.
w: The weight matrix.
b: The bias vector.
h: The list of outputs h of the LSTM.
states: The list of states (it is the concatenated vector of c an h).
)doc");

REGISTER_OP("LSTMBlockGrad")
    .Attr("cell_size: int")
    .Attr("sequence_len_max: int")
    .Attr("bprop_dx: bool = true")
    .Input("sequence_len: int64")
    .Input("initial_state: float")
    .Input("x: sequence_len_max * float")
    .Input("w: float")
    .Input("b: float")
    .Input("i: sequence_len_max * float")
    .Input("cs: sequence_len_max * float")
    .Input("f: sequence_len_max * float")
    .Input("o: sequence_len_max * float")
    .Input("ci: sequence_len_max * float")
    .Input("co: sequence_len_max * float")
    .Input("states: sequence_len_max * float")
    .Input("h: sequence_len_max * float")
    .Input("h_grad: sequence_len_max * float")
    .Output("x_grad: sequence_len_max * float")
    .Output("w_grad: float")
    .Output("b_grad: float")
    .Doc(R"doc(
Computes the LSTM backward propagation for N time steps.

This implementation is to be used inconjunction of LSTMBlock.

cell_size: The LSTM cell size.
forget_bias: The forget gate bias.
sequence_len: A vector of batch_size containing the sequence length.
initial_state: Initial state of the LSTM.
x: The list of inputs to the LSTM.
w: The weight matrix.
b: The bias vector.
h: The list of outputs h of the LSTM.
states: The list of states (it is the concatenated vector of c an h).
x_grad: The list of grads for x.
w_grad: The grad for w.
b_grad: The grad for b.
)doc");

}  // end namespace tensorflow
