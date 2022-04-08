# BERT Inference Using TensorRT

Original README.md is https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/README.md.

# Quick Setup

1. Git clone the repository

```bash
git clone https://github.com/leo0519/TensorRT.git --branch cusparselt-plugin && cd TensorRT
```

2. Make sure the cuSPARSELt tgz file is inside this directory

```bash
ls libcusparse_lt_0.3.0.1_test.tar.gz
```

3. Build docker environment

```bash
./docker/build.sh --file docker/ubuntu-18.04.Dockerfile --tag tensorrt-plugin-ubuntu18.04-cuda11.4
```

4. launch docker environment

```bash
./docker/launch.sh --tag tensorrt-plugin-ubuntu18.04-cuda11.4 --gpus all
```

5. Change directory

```bash
cd TensorRT/demo/BERT
```

6. Compile plugin shared library

```bash
cd plugin && rm -rf build && mkdir build && cd build
cmake .. && make -j4
cd ../..
```

7. Download the model and data

```bash
bash ./scripts/download_squad.sh
bash ./scripts/download_model.sh tf large v2 384
bash ./scripts/download_model.sh pyt megatron-large int8-qat sparse
```

8. Run the model

```bash
bash build_infer.sh            # Baseline
bash build_infer.sh condition1 # Condition1
bash build_infer.sh condition2 # Condition2
```

9. The results of accuracy

```bash
{"exact_match": 84.15, "f1": 91.03}   # Baseline
{"exact_match": 84.00, "f1": 84.00}   # Condition1
{"exact_match": 90.86, "f1": 90.87}   # Condition2
```
