from concurrent.futures import ThreadPoolExecutor

import numpy.random
from openvino.runtime import Core
from tqdm import tqdm

N_ITER = 1000
N_STREAM = 24  # Core number of 13700K

ie = Core()
ie.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
model_xml = f"outputs/model/resnet_50/openvino/int8/model.xml"
compiled_model = ie.compile_model(model_xml, "CPU")

with tqdm(unit="frame") as pbar:
    input = numpy.random.rand(1, 3, 224, 224)


    def infer_stream():
        outputs = []
        infer_req = compiled_model.create_infer_request()
        for _ in range(N_ITER):
            infer_req.infer(input)
            pbar.update(1)
        return outputs


    with ThreadPoolExecutor(N_STREAM) as pool:
        tasks = []
        for _ in range(N_STREAM):
            tasks.append(pool.submit(infer_stream))
        outputs = [task.result() for task in tasks]
