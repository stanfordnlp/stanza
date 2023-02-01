"""
Utility methods to check that all processors are on the expected device

Refactored since it can be used for multiple pipelines
"""

import warnings

import torch

def check_on_gpu(pipeline):
    """
    Check that the processors are all on the GPU and that basic execution works
    """
    if not torch.cuda.is_available():
        warnings.warn("Unable to run the test that checks the pipeline is on the GPU, as there is no GPU available!")
        return

    for name, proc in pipeline.processors.items():
        if proc.trainer is not None:
            device = next(proc.trainer.model.parameters()).device
        else:
            device = next(proc._model.parameters()).device

        assert str(device).startswith("cuda"), "Processor %s was not on the GPU" % name

    # just check that there are no cpu/cuda tensor conflicts
    # when running on the GPU
    pipeline("This is a small test")

def check_on_cpu(pipeline):
    """
    Check that the processors are all on the CPU and that basic execution works
    """
    for name, proc in pipeline.processors.items():
        if proc.trainer is not None:
            device = next(proc.trainer.model.parameters()).device
        else:
            device = next(proc._model.parameters()).device

        assert str(device).startswith("cpu"), "Processor %s was not on the CPU" % name

    # just check that there are no cpu/cuda tensor conflicts
    # when running on the CPU
    pipeline("This is a small test")
