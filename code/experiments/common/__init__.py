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
``real_cells``
    The ``--real`` option: a checkpoint's integer graph and prompts in place of random weights, and the rule that
    decides which of the paper's cells take a real checkpoint.
``tampers``
    The trace-tampering attacks every timed decoder cell also runs.
``client``
    The CPU client in a process of its own, for the client's memory.
``energy``
    The GPU's energy counter (NVML), for the energy per query.
``stats``
    Seeded bootstrap intervals of the reported medians, ratios and differences.
"""
