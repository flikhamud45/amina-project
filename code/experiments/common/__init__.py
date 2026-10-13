"""What the experiment scripts share.

``paths``
    Where the stored models, records and caches live.
``cli``
    The protocol configuration of the measured results as command-line flags, and the runtime flags (devices,
    threads, implementation options).
``records``
    The run environment for a record, peak memory, timing on a CPU or a GPU, JSON output.
``models``
    The benchmark models: the small decoder shapes of the tests and rehearsals, and the trained image models.
``checkpoints``
    Real checkpoints: OPT and Llama-family models in fp32, WikiText-2 windows, the certified models' calibration and
    their integer graphs.
``tampers``
    The trace-tampering attacks every timed decoder cell also runs.
"""
