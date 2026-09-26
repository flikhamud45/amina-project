#!/bin/bash
# Portable: set PROJECT_DIR to your checkout, or rely on the default.
PROJECT_DIR="${PROJECT_DIR:-$HOME/amina-project}"
#SBATCH --partition=gpu-bermano
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=batched_scale_job-%j.out

VENV=${PROJECT_DIR}/.venv
cd ${PROJECT_DIR}/code

"$VENV/bin/python" - <<'PY'
import time
import numpy as np
from pvi.training import load_network
from pvi.zoo import mlp_architecture_large_for
from pvi.data import load_classification
from pvi.attacks.backdoor import calibrate_activation_ceilings
from pvi.attacks.tamper import apply_plan, plan_single_neuron_flip
from pvi.defences import LocalContributionSampler
from pvi.defences.adaptive import plan_adaptive_stealthy_flip
from pvi.experiments.analysis import acceptance_probability
from pvi.protocol import ProtocolParams
from pvi.protocol.batched import (
    BatchedWeightCommitment, fixed_point_layer, prove_layer, verify_layer,
)

LAYER, SCALE, NQ, QUERIES = 1, 8, 24, 10
net = load_network(mlp_architecture_large_for(10), "artifacts/models/mlp_mnist_large.npz")
ds = load_classification(name="mnist", seed=0)
train_x = ds.train_x.reshape(len(ds.train_x), -1)
test_x = ds.test_x.reshape(len(ds.test_x), -1)
params = ProtocolParams()
layer = net.architecture[LAYER]
W, b = net.parameters[layer.name]
width = layer.n_neurons
print(f"layer {LAYER}, width {width}", flush=True)

ref = fixed_point_layer(W, b, net.eval_trace(test_x[0])[LAYER - 1], scale_bits=SCALE)
z = ref.pre_activations()
print(f"z range {int(z.min())} .. {int(z.max())}, field half {ref.prime//2}", flush=True)
t0 = time.perf_counter()
com = BatchedWeightCommitment(ref.matrix, prime=ref.prime)
print(f"commitment built in {time.perf_counter()-t0:.1f}s, {com.n_columns} columns", flush=True)

def honest_view(trace):
    return fixed_point_layer(W, b, trace[LAYER-1], scale_bits=SCALE)

def check(view):
    p = prove_layer(view, com, n_queries=NQ)
    ok = verify_layer(p, com.digest, com.params, com, view.inputs, view.outputs,
                      prime=view.prime, n_queries=NQ)
    return ok, p

acc, sizes = 0, []
for q in test_x[:QUERIES]:
    ok, p = check(honest_view(net.eval_trace(q)))
    acc += ok; sizes.append(p.size_bytes())
print(f"honest completeness: {acc}/{QUERIES}, mean proof {np.mean(sizes)/1024:.1f} kB", flush=True)

ceil = calibrate_activation_ceilings(net, train_x[:3000], layer_index=LAYER, percentile=100.0)
samp = LocalContributionSampler(net)
for kind in ("naive", "zero_hiding"):
    caught = total = 0; unif = []
    for q in test_x[:QUERIES]:
        tr = net.eval_trace(q)
        if kind == "naive":
            plan = plan_single_neuron_flip(net, tr, layer_index=LAYER)
        else:
            s = plan_adaptive_stealthy_flip(net, tr, layer_index=LAYER, ceilings=ceil,
                                            sampler=samp, params=params)
            plan = s.plan if s.succeeded else None
        if plan is None:
            continue
        forged = apply_plan(net, tr, plan)
        if int(forged.output.argmax()) == int(tr.output.argmax()):
            continue
        total += 1
        hv = honest_view(tr)
        outs = np.maximum(np.array(hv.pre_activations(), dtype=np.int64), 0)
        sc = float(1 << (2*SCALE))
        for n, v in zip(plan.neurons, plan.new_values):
            outs[int(n)] = int(round(float(v) * sc))
        view = fixed_point_layer(W, b, tr[LAYER-1], activations_out=outs, scale_bits=SCALE)
        ok, _ = check(view)
        caught += not ok
        unif.append(1.0 - acceptance_probability(net, forged, params))
    print(f"{kind}: batched detection {caught}/{total} = {caught/max(total,1):.6f}, "
          f"uniform sampling {np.mean(unif):.6f}", flush=True)
print("BATCHED_SCALE_DONE")
PY
