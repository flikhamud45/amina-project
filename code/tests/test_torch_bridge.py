"""The PyTorch module and the traced network must compute the same function.

Training happens in PyTorch; committing and tracing happen in NumPy.  If the two
drifted apart, we would be training one model and publishing a commitment to
another -- a silent, and rather embarrassing, form of exactly the substitution the
protocol is meant to detect.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pvi.nn.models import (
    build_torch_module,
    extract_parameters,
    mlp_architecture,
    small_cnn_architecture,
)
from pvi.nn.network import TracedNetwork


@pytest.mark.parametrize(
    "architecture",
    [
        mlp_architecture(24, [32, 16], 5),
        small_cnn_architecture((1, 12, 12), 4, channels=(4, 6), hidden_width=16),
        small_cnn_architecture((3, 10, 10), 3, channels=(6, 4), hidden_width=12),
    ],
    ids=["mlp", "cnn-1ch", "cnn-3ch"],
)
def test_torch_and_traced_network_agree(architecture):
    torch.manual_seed(0)
    module = build_torch_module(architecture)
    module.eval()
    network = TracedNetwork(architecture, extract_parameters(module, architecture))

    rng = np.random.default_rng(0)
    batch = rng.standard_normal((16, architecture.input_layer.n_neurons)).astype(np.float32)

    with torch.no_grad():
        expected = module(torch.from_numpy(batch)).numpy()
    actual = network.forward_batch(batch)[-1]

    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-4


def test_extracted_parameters_match_module_state():
    architecture = mlp_architecture(10, [8], 3)
    torch.manual_seed(1)
    module = build_torch_module(architecture)
    parameters = extract_parameters(module, architecture)
    for name, (weight, bias) in parameters.items():
        assert np.array_equal(weight, module.layers[name].weight.detach().numpy())
        assert np.array_equal(bias, module.layers[name].bias.detach().numpy())


def test_predictions_agree_between_backends():
    architecture = small_cnn_architecture((1, 12, 12), 4, channels=(4,), hidden_width=16)
    torch.manual_seed(2)
    module = build_torch_module(architecture)
    module.eval()
    network = TracedNetwork(architecture, extract_parameters(module, architecture))

    rng = np.random.default_rng(2)
    batch = rng.standard_normal((64, architecture.input_layer.n_neurons)).astype(np.float32)
    with torch.no_grad():
        torch_pred = module(torch.from_numpy(batch)).argmax(dim=1).numpy()
    assert np.array_equal(network.predict(batch), torch_pred)


def test_eval_trace_final_layer_is_the_model_output():
    architecture = mlp_architecture(10, [8, 6], 3)
    torch.manual_seed(3)
    module = build_torch_module(architecture)
    network = TracedNetwork(architecture, extract_parameters(module, architecture))
    rng = np.random.default_rng(3)
    query = rng.standard_normal(10).astype(np.float32)
    trace = network.eval_trace(query)
    assert np.allclose(trace.output, network.forward_batch(query[None, :])[-1][0])


def test_saving_and_loading_round_trips(tmp_path):
    from pvi.training import load_network, save_network

    architecture = mlp_architecture(10, [8], 3)
    torch.manual_seed(4)
    module = build_torch_module(architecture)
    network = TracedNetwork(architecture, extract_parameters(module, architecture))
    path = tmp_path / "model.npz"
    save_network(network, path)
    restored = load_network(architecture, path)

    rng = np.random.default_rng(4)
    batch = rng.standard_normal((8, 10)).astype(np.float32)
    assert np.array_equal(network.forward_batch(batch)[-1], restored.forward_batch(batch)[-1])


def test_traced_network_rejects_extra_parameters():
    architecture = mlp_architecture(6, [4], 2)
    torch.manual_seed(5)
    parameters = dict(extract_parameters(build_torch_module(architecture), architecture))
    parameters["input"] = (np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.float32))
    with pytest.raises(ValueError):
        TracedNetwork(architecture, parameters)
