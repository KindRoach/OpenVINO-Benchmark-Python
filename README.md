# OpenVINO Benchmark Python

Benchmark in python code for a classical computer vision application use case: "Decode &amp; Inference".

## Prepare Test Data and Models

```bash
python prepare_data_and_model.py -m all
```

## Run Video Decode Benchmark

```bash
python run_decode.py
```

You could use `python run_decode.py -h` to learn about benchmark options.

## Run OpemVINO Inference Benchmark

```bash
python run_infer.py
```

You could use `python run_infer.py -h` to learn about benchmark options.
