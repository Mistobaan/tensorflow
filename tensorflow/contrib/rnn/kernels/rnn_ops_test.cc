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

#include <functional>
#include <memory>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {
namespace {

class RNNOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(
        NodeDefBuilder("myop", "RNN")
            .Input(FakeInput())
            .Input(FakeInput())
            .Input(FakeInput({DT_BOOL, DT_INT32, DT_FLOAT, DT_DOUBLE, DT_QINT8,
                              DT_QINT32, DT_UINT8, DT_INT8, DT_INT16, DT_INT64,
                              DT_STRING, DT_COMPLEX64}))
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RNNOpTest, Simple) {
}

} // namespace
} // namespace tensorflow
