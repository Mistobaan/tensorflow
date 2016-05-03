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

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("LSTMBlock")
    .Attr("cell_size: int")
    .Attr("forget_bias: float = 1.0")
    .Attr("sequence_len_max: int")
    .Input("sequence_len: int64")
    .Input("initial_state: float")
    .Input("x: sequence_len_max * float")
    .Input("w: float")
    .Input("b: float")
    .Output("y: sequence_len_max * float")
    .Output("h: sequence_len_max * float")
    .Output("c: sequence_len_max * float")
    .Doc(R"doc(
)doc");

// REGISTER_OP("LSTMBlockGrad")
//     .Attr("cell_size: int")
//     .Attr("sequence_len_max: int")
//     .Input("sequence_len: int64")
//     .Input("initial_state: float")
//     .Input("x: sequence_len_max * float")
//     .Input("w: float")
//     .Input("b: float")
//     .Input("states: sequence_len_max * float")
//     .Input("h_grad: sequence_len_max * float")
//     .Output("x_grad: sequence_len_max * float")
//     .Output("w_grad: float")
//     .Output("b_grad: float")
//     .Doc(R"doc(
// )doc");

}  // end namespace tensorflow
