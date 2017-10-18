exports_files([
  "configure.py", 
])

py_binary(
    name = "configure",
    srcs = [
        "configure.py",
    ],
    data = [
        "//tensorflow/tools/git:gen_git_source.py",
    ],
)
