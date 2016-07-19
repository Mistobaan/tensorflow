package(
    default_visibility = ["//visibility:public"],
)

load("@io_bazel_rules_go//go:def.bzl", "go_prefix", "go_library", "go_test")
load("//tensorflow/tools/go:build_config.bzl", "go_wrap_cc", "go_proto_library")

go_prefix("github.com/tensorflow/tensorflow/contrib/go")

go_proto_library(
    name = "tensorflow_protos_all_go",
    srcs = ["//tensoflow/core:protos_all"],
    go_libs = ["@go_protobuf//:proto"],
    protoc_gen_go = "@go_protobuf//:protoc-gen-go-bin",
    deps = [],
)

go_wrap_cc(
    name = "go_wrap_tensorflow",
    srcs = ["tensorflow.swigcxx"],
    go_module_name = "tensorflow/proto",
    module_name = "tensorflow",
    swig_includes = [],
    deps = [
        "//tensorflow:libtensorflow.so",
        #      "//tensorflow/core:protos_all_go",
        #        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        #        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        #        "//tensorflow/core:lib",
        #        "//tensorflow/core/distributed_runtime:server_lib",
    ],
)

# go_library(
#     name = "go_default_library",
#     srcs = [":all_files"],
#     library = "//tensorflow:libtensorflow.so",
# )

# go_test(
#     name = "tf_golib_test",
#     srcs = [":all_files"],
#     library = ":go_default_library"
# )

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*.go"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

# Documentation.
genrule(
    name = "generate_docs",
    srcs = [":all_files"],
    outs = [
        "g3doc/tensor.md",
        "g3doc/session.md",
        "g3doc/graph.md",
    ],
    cmd = """bash -c `
    for name in Tensor, Session, Graph; do
        godoc  -ex=true -templates=godoc_tmpl/ github.com/tensorflow/tensorflow/tensorflow/contrib/go $$name > g3doc/$$name.md
    endfor;
    `
    """,
    visibility = ["//visibility:private"],
)

# go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ github.com/tensorflow/tensorflow/tensorflow/contrib/go Tensor > g3doc/tensor.md"
# go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ github.com/tensorflow/tensorflow/tensorflow/contrib/go Session > g3doc/session.md"
# go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ github.com/tensorflow/tensorflow/tensorflow/contrib/go Graph > g3doc/graph.md"
