# BERT Inference Using TensorRT

Original README.md is https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/README.md.

# Quick Setup

1. Git clone the repository

```bash
git clone https://github.com/leo0519/TensorRT.git --branch cusparselt-plugin && cd TensorRT
```

2. Build docker environment

```bash
./docker/build.sh --file docker/ubuntu-18.04.Dockerfile --tag tensorrt-plugin-ubuntu18.04-cuda11.4
```

3. launch docker environment

```bash
./docker/launch.sh --tag tensorrt-plugin-ubuntu18.04-cuda11.4 --gpus all
```

4. Change directory

```bash
cd TensorRT/demo/BERT
```

5. Compile plugin shared library

```bash
cd plugin && rm -rf build && mkdir build && cd build
cmake .. && make -j4
cd ../..
```

6. Download the model and data

```bash
bash ./scripts/download_squad.sh
bash ./scripts/download_model.sh tf large v2 384
bash ./scripts/download_model.sh pyt megatron-large int8-qat sparse
```

7. Run the model

```bash
bash build_infer.sh            # Baseline
bash build_infer.sh condition1 # Condition1
bash build_infer.sh condition2 # Condition2
```

8. The results of accuracy

```bash
{"exact_match": 83.9829706717124, "f1": 90.8383200226445}   # Baseline
{"exact_match": 83.85998107852413, "f1": 90.83975654462425} # Condition1
{"exact_match": 81.22989593188268, "f1": 88.91073628832046} # Condition2
```
