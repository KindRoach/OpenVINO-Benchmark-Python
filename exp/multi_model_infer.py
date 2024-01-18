import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import tqdm
from openvino import Core, properties, CompiledModel
from openvino._pyopenvino.properties.hint import SchedulingCoreType
from tqdm import tqdm

from exp.exp_util import loop_seconds
from utils import OV_MODEL_PATH_PATTERN


def print_ov_devices_properties(core):
    print("All OpenVINO supported devices are:")

    available_devices = core.available_devices
    for device in available_devices:
        full_name = core.get_property(device, properties.device.full_name())
        print(full_name)

        for p in [
            properties.enable_profiling,
            properties.hint.inference_precision,
            properties.hint.performance_mode,
            properties.hint.execution_mode,
            properties.hint.num_requests,
            properties.hint.scheduling_core_type,
            properties.hint.enable_hyper_threading,
            properties.hint.enable_cpu_pinning,
            properties.num_streams,
            properties.affinity,
            properties.inference_num_threads,
            properties.cache_dir,
            properties.intel_cpu.denormals_optimization,
            properties.intel_cpu.sparse_weights_decompression_rate,
        ]:
            try:
                v = core.get_property(device, p())
                print(f"\t{p()} = {v}")
            except RuntimeError as e:
                pass

    print("----------------------------------")


def infer_model(model: CompiledModel, inputs: np.ndarray):
    _ = model(inputs)


def cool_system():
    time.sleep(10)


N = 10000
SECONDS = 15


def sync_infer(models: List[CompiledModel]):
    inputs = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for tensor in tqdm(loop_seconds(SECONDS, inputs), desc="Sync"):
        for m in models:
            infer_model(m, tensor)


def async_infer(models: List[CompiledModel]):
    with ThreadPoolExecutor() as pool:
        inputs = np.random.rand(1, 3, 224, 224).astype(np.float32)
        for tensor in tqdm(loop_seconds(SECONDS, inputs), desc="Async"):
            futures = [pool.submit(infer_model, m, tensor) for m in models]
            rets = [f.result() for f in futures]


def exp(models: List[CompiledModel]):
    sync_infer(models)
    async_infer(models)


def all_p_core_differ_model():
    print("all_p_core_differ_model:")

    core = Core()
    res18 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU")
    res50 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet50", "int8"), "CPU")
    res101 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet101", "int8"), "CPU")

    exp([res18, res50, res101])


def all_p_core_same_model():
    print("all_p_core_same_model:")

    core = Core()
    res18_a = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU")
    res18_b = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU")
    res18_c = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU")

    exp([res18_a, res18_b, res18_c])


P_CORE_CONFIG = {properties.hint.scheduling_core_type(): SchedulingCoreType.PCORE_ONLY}
E_CORE_CONFIG = {properties.hint.scheduling_core_type(): SchedulingCoreType.ECORE_ONLY}


def p_e_core_differ_model():
    print("p_e_core_differ_model:")

    core = Core()
    res18 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU", E_CORE_CONFIG)
    res50 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet50", "int8"), "CPU", P_CORE_CONFIG)
    res101 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet101", "int8"), "CPU", P_CORE_CONFIG)

    exp([res18, res50, res101])


def p_e_core_same_model():
    print("p_e_core_same_model:")

    core = Core()
    res18_a = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU", E_CORE_CONFIG)
    res18_b = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU", P_CORE_CONFIG)
    res18_c = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), "CPU", P_CORE_CONFIG)

    exp([res18_a, res18_b, res18_c])


GPU_DEVICE = "GPU.0"


def gpu_differ_model():
    print("gpu_differ_model:")

    core = Core()
    res18 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), GPU_DEVICE)
    res50 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet50", "int8"), GPU_DEVICE)
    res101 = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet101", "int8"), GPU_DEVICE)

    exp([res18, res50, res101])


def gpu_same_model():
    print("gpu_same_model:")

    core = Core()
    res18_a = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), GPU_DEVICE)
    res18_b = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), GPU_DEVICE)
    res18_c = core.compile_model(OV_MODEL_PATH_PATTERN % ("resnet18", "int8"), GPU_DEVICE)

    exp([res18_a, res18_b, res18_c])


def main():
    core = Core()
    print_ov_devices_properties(core)

    all_p_core_differ_model()
    cool_system()

    p_e_core_differ_model()
    cool_system()

    gpu_differ_model()
    cool_system()

    all_p_core_same_model()
    cool_system()

    p_e_core_same_model()
    cool_system()

    gpu_same_model()
    cool_system()


if __name__ == '__main__':
    main()
