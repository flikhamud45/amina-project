### Models: accuracy of the float model and of the exact int8 model the protocol verifies

| model | params | float acc. | int8 acc. | agreement | claimed values / query |
|---|---|---|---|---|---|
| MLP (MNIST) | 536K | 97.72% | 97.71% | 99.99% | 778 |
| LeNet-5 (MNIST) | 62K | 99.33% | 99.34% | 99.99% | 6,518 |
| VGG-11 (CIFAR-10) | 9.8M | 91.05% | 91.09% | 99.45% | 152,586 |
| VGG-16 (CIFAR-10) | 15.2M | 93.14% | 93.03% | 99.63% | 277,514 |
| ResNet-18 (CIFAR-10) | 11.2M | 94.18% | 94.21% | 99.71% | 614,410 |
| ResNet-18 (224px, dogs vs cats) | 11.2M | 94.60% | 95.00% | 99.60% | 2,483,714 |

### CNNs, one query, λ=128 (prover RTX 2080 Ti, verifier Xeon 4114 x8): ours vs downloading the model

| model | prove (C) | verify (C) | proof (C) | proof (C), zlib | verify (Kpre) | proof (Kpre) | int8 model | CPU re-run |
|---|---|---|---|---|---|---|---|---|
| MLP (MNIST) | 10.9 ms | 21.4 ms | 270 kB | 269 kB | 3.7 ms | 3.11 kB | 538 kB | 225 µs |
| LeNet-5 (MNIST) | 16.3 ms | 28.8 ms | 131 kB | 116 kB | 8.4 ms | 26.1 kB | 62.4 kB | 1.7 ms |
| VGG-11 (CIFAR-10) | 64.8 ms | 134.1 ms | 2.2 MB | 2.01 MB | 29.7 ms | 610 kB | 9.76 MB | 5.6 ms |
| VGG-16 (CIFAR-10) | 90.7 ms | 194.1 ms | 3.43 MB | 3.09 MB | 52.0 ms | 1.11 MB | 15.3 MB | 9.0 ms |
| ResNet-18 (CIFAR-10) | 69.1 ms | 180.3 ms | 4.63 MB | 3.86 MB | 75.2 ms | 2.46 MB | 11.2 MB | 10.4 ms |
| ResNet-18 (224px, dogs vs cats) | 105.3 ms | 369.7 ms | 12.1 MB | 9.13 MB | 223.8 ms | 9.93 MB | 11.2 MB | 26.0 ms |

### Anchuri et al.'s path test on the same models: cost of reaching 2^-40 against the single-neuron attack

| model | 1 path | verify 1 path | detect / path | paths for 2^-40 | k paths, shared | open everything | ours at λ=40 (C) |
|---|---|---|---|---|---|---|---|
| MLP (MNIST) | 3.06 kB | 308 µs | 1/256 | 7,084 | 542 kB | 542 kB | 97.5 kB |
| LeNet-5 (MNIST) | 3.91 kB | 1.9 ms | 1/84 | 2,316 | 72.4 kB | 71.5 kB | 67.2 kB |
| VGG-11 (CIFAR-10) | 47 kB | 1.7 ms | 1/512 | 14,182 | 9.96 MB | 9.96 MB | 1.19 MB |
| VGG-16 (CIFAR-10) | 71.7 kB | 5.7 ms | 1/512 | 14,182 | 15.6 MB | 15.6 MB | 1.97 MB |
| ResNet-18 (CIFAR-10) | 76.9 kB | 3.1 ms | 1/512 | 14,182 | 11.8 MB | 11.8 MB | 3.26 MB |
| ResNet-18 (224px, dogs vs cats) | 101 kB | 8.1 ms | 1/512 | 14,182 | 13.7 MB | 13.7 MB | 10.7 MB |

### Detection. [1]: per-path probability of catching a single tampered neuron (exact); ours: attacks rejected / attempted (λ=40, committed weights)

| model | [1] attack neuron | [1] worst neuron | single value | penult. neuron | logit | 1% weights | M~ model | forged fold | forged column |
|---|---|---|---|---|---|---|---|---|---|
| MLP (MNIST) | 1/256 | 1/512 | 100/100 | 100/100 | 100/100 | 10/10 | – | 10/10 | 10/10 |
| LeNet-5 (MNIST) | 1/84 | 1/60,000 | 100/100 | 100/100 | 100/100 | 10/10 | – | 10/10 | 10/10 |
| VGG-11 (CIFAR-10) | 1/512 | 1/197,001 | 100/100 | 100/100 | 100/100 | 10/10 | – | 10/10 | 10/10 |
| VGG-16 (CIFAR-10) | 1/512 | 1/262,670 | 100/100 | 100/100 | 100/100 | 10/10 | – | 10/10 | 10/10 |
| ResNet-18 (CIFAR-10) | 1/512 | 1/1,492,601 | 100/100 | 100/100 | 100/100 | 10/10 | – | 10/10 | 10/10 |
| ResNet-18 (224px, dogs vs cats) | 1/512 | 1/27,974,105 | 60/60 | 60/60 | 60/60 | 6/6 | 6/6 | 6/6 | 6/6 |

### The single-neuron attack on the float model (one penultimate neuron set to a large value)

| model | penultimate width | flips (any neuron) | flips (live neurons) | value / neuron's max (median) |
|---|---|---|---|---|
| MLP (MNIST) | 256 | 100.00% | 100.00% | 5.3× |
| LeNet-5 (MNIST) | 84 | 100.00% | 100.00% | 2.2× |
| VGG-11 (CIFAR-10) | 512 | 100.00% | 100.00% | 40× |
| VGG-16 (CIFAR-10) | 512 | 100.00% | 100.00% | 48× |
| ResNet-18 (CIFAR-10) | 512 | 100.00% | 100.00% | 13× |
| ResNet-18 (224px, dogs vs cats) | 512 | 100.00% | 100.00% | 9.7× |

### Language models, one prompt (next-token logits), λ=128

| model | params | tokens | prove (C) | verify (C) | proof (C) | verify (Kpre) | proof (Kpre) | source |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | 124.4M | 64 | 439.0 ms | 985.1 ms | 62.1 MB | 279.3 ms | 21.8 MB | measured |
| GPT-2 | 124.4M | 128 | 481.5 ms | 1.09 s | 83.7 MB | 410.9 ms | 43.5 MB | measured |
| GPT-2 | 124.4M | 256 | 489.5 ms | 1.27 s | 127 MB | 561.8 ms | 86.7 MB | measured |
| GPT-2 | 124.4M | 512 | 636.8 ms | 2.13 s | 213 MB | 1.41 s | 173 MB | measured |
| OPT-125M | 125.2M | 64 | 687.4 ms | 1.47 s | 62.1 MB | 381.3 ms | 21.8 MB | measured |
| OPT-125M | 125.2M | 2048 | 1.50 s | 22.14 s | 733 MB | 21.27 s | 692 MB | measured |
| OPT-1.3B | 1.32B | 64 | 2.91 s | 3.50 s | 264 MB | 1.28 s | 114 MB | extrap. |
| OPT-1.3B | 1.32B | 2048 | 13.03 s | 2.1 min | 3.81 GB | 2.1 min | 3.66 GB | extrap. |
| Qwen3-4B | 4.02B | 8 | 12.65 s | 6.08 s | 410 MB | 1.06 s | 36.1 MB | extrap. |
| Qwen3-4B | 4.02B | 64 | 10.00 s | 6.63 s | 659 MB | 2.16 s | 284 MB | extrap. |
| OPT-6.7B | 6.66B | 64 | 9.45 s | 8.21 s | 674 MB | 2.89 s | 304 MB | extrap. |
| OPT-6.7B | 6.66B | 2048 | 49.89 s | 3.4 min | 10.1 GB | 3.3 min | 9.73 GB | extrap. |
| Llama-2-7B | 6.74B | 64 | 11.54 s | 7.17 s | 762 MB | 2.72 s | 349 MB | extrap. |
| Llama-2-7B | 6.74B | 512 | 14.15 s | 30.63 s | 3.21 GB | 27.88 s | 2.79 GB | extrap. |
| Llama-2-7B | 6.74B | 2048 | 32.45 s | 3.7 min | 11.6 GB | 3.5 min | 11.2 GB | extrap. |
| Llama-2-13B | 13.02B | 64 | 19.01 s | 15.50 s | 1.19 GB | 5.64 s | 547 MB | extrap. |

### Published results used in the comparison (as reported; hardware differs per row)

| system | model | params | tokens | prove | verify | proof | hardware | source |
|---|---|---|---|---|---|---|---|---|
| zkCNN | LeNet-5 MNIST | 62K | – | 441.0 ms | 5.8 ms | 71.3 kB | AMD EPYC 7R32, 1 core | ePrint 2021/673, Table 1, p.13; accuracy |
| zkCNN | VGG-11 CIFAR-10 | 9.8M | – | 47.80 s | 39.3 ms | 304 kB | AMD EPYC 7R32, 1 core | ePrint 2021/673, Table 1, p.13; accuracy |
| zkCNN | VGG-16 CIFAR-10 | 15.2M | – | 88.30 s | 59.3 ms | 341 kB | AMD EPYC 7R32, 1 core | ePrint 2021/673, Abstract p.1 and Table  |
| vCNN (re-run by zkCNN) | LeNet-5 MNIST | 62K | – | 5.49 s | 84.0 ms | 340 B | AMD EPYC 7R32 | ePrint 2021/673, Table 2, p.13 |
| ZKML | ResNet-18 CIFAR-10 (ZKML's 281K-param variant) | 281K | – | 52.90 s | 11.8 ms | 15.7 kB | AWS r6i.8xlarge, 32 vCPU | Table 6, p.11; Table 8, p.11 |
| ZKML | VGG-16 CIFAR-10 | 15.2M | – | 10.6 min | 9.6 ms | 12.1 kB | AWS r6i.8xlarge, 32 vCPU | Table 6, p.11; Tables 8-9, p.11-12 |
| ZKML | MobileNet v2 ImageNet | 3.5M | – | 20.4 min | 17.7 ms | 17.7 kB | AWS r6i.16xlarge, 64 vCPU | Table 6, p.11 |
| ZKML | GPT-2 (distilled) | 81.3M | – | 60.9 min | 18.70 s | 28.1 kB | AWS r6i.32xlarge, 128 vCPU | Table 6, p.11 |
| ZENO | VGG-16 CIFAR-10 (19.9M-FLOP variant) | – | – | 48.00 s | – | – | Xeon Gold 5218 | ASPLOS'24 camera-ready, Table 5 p.12; te |
| ZENO | ResNet-18 CIFAR-10 (32.4M-FLOP variant) | – | – | 102.00 s | – | – | Xeon Gold 5218 | ASPLOS'24 camera-ready, Table 5 p.12; Ta |
| Bionetta | MNIST MLP (2M) | 2.0M | – | 3.05 s | 10.0 ms | 880 B | Xeon E5-2665, 16 threads | Table 4, p.25 |
| Bionetta | LeNet-5 | – | – | 3.75 s | 10.0 ms | 880 B | Xeon E5-2665, 16 threads | Table 4, p.25 |
| Bionetta | ResNet-18 (Bionetta's version) | – | – | 14.10 s | 15.0 ms | 880 B | Xeon E5-2665, 16 threads | Table 4, p.25 |
| EZKL (run by Bionetta) | MNIST MLP (2M) | 2.0M | – | 21.8 min | 5.40 s | 127 kB | Xeon E5-2665, 16 threads | Bionetta arXiv 2510.06784v2, Table 4, p. |
| EZKL (run by Bionetta) | LeNet-5 | – | – | 8.9 min | 2.00 s | 127 kB | Xeon E5-2665, 16 threads | Bionetta Table 4, p.25 |
| EZKL (run by Bionetta) | ResNet-18 (Bionetta's version) | – | – | 114.0 min | 25.30 s | 180 kB | Xeon E5-2665, 16 threads | Bionetta Table 4, p.25 |
| ZEN | LeNet-5-small CIFAR-10 | – | – | 2.1 min | 23.0 ms | 192 B | Azure M32ls, 32 cores | ePrint 2021/087, Table 1 p.4; Table 6 p. |
| Mystique (interactive) | ResNet-50 CIFAR-10 | 23.5M | – | 2.6 min | – | 530 MB | 2x AWS m5.2xlarge, 200 Mbps | ePrint 2021/730, Table 3, p.23 |
| zkPyTorch | VGG-16 CIFAR-10 | 15.2M | – | 2.20 s | – | – | single CPU core | ePrint 2025/535, Table 1 p.2; Table 2 p. |
| zkLLM | OPT-125M | 125.0M | 2048 | 73.90 s | 342.0 ms | 141 kB | A100 40GB | Table 1, p.12 (setup p.11) |
| zkLLM | OPT-350M | 350.0M | 2048 | 111.00 s | 593.0 ms | 144 kB | A100 40GB | Table 1, p.12 |
| zkLLM | OPT-1.3B | 1.30B | 2048 | 3.7 min | 899.0 ms | 147 kB | A100 40GB | Table 1, p.12 |
| zkLLM | OPT-2.7B | 2.70B | 2048 | 5.9 min | 1.41 s | 152 kB | A100 40GB | Table 1, p.12 |
| zkLLM | OPT-6.7B | 6.70B | 2048 | 9.1 min | 2.08 s | 157 kB | A100 40GB | Table 1, p.12 |
| zkLLM | OPT-13B | 13.00B | 2048 | 11.9 min | 3.71 s | 160 kB | A100 40GB | Table 1, p.12 |
| zkLLM | Llama-2-7B | 6.74B | 2048 | 10.3 min | 2.36 s | 183 kB | A100 40GB | Table 1, p.12 |
| zkLLM | Llama-2-13B | 13.00B | 2048 | 13.4 min | 3.95 s | 188 kB | A100 40GB | Table 1, p.12 |
| zkLLM (re-run by Anchuri) | Llama-2-7B | 6.74B | – | 6.5 min | – | – | RTX 3090 | Anchuri et al., Sec 8.1 p.27 (setup) and |
| zkLLM (re-run by zkGPT) | GPT-2 | 124.0M | – | 15.80 s | 540.0 ms | 126 kB | A100 | zkGPT (USENIX Sec 2025), Table 3, p.2056 |
| zkGPT | GPT-2 | 124.0M | – | 21.80 s | 350.0 ms | 101 kB | Xeon 6126, 32 threads | Table 3, p.2056 (PDF p.13); hardware Sec |
| DeepProve | GPT-2 | 124.0M | 64 | 34.20 s | 1.35 s | 21.7 MB | Ryzen 9 7950X3D, 16 cores | Table 2, p.11 |
| DeepProve | GPT-2 | 124.0M | 128 | 51.00 s | 1.45 s | 22.9 MB | Ryzen 9 7950X3D, 16 cores | Table 2, p.11 |
| DeepProve | GPT-2 | 124.0M | 256 | 87.60 s | 1.55 s | 24.1 MB | Ryzen 9 7950X3D, 16 cores | Table 2, p.11 |
| DeepProve | GPT-2 | 124.0M | 512 | 2.9 min | 1.65 s | 25.5 MB | Ryzen 9 7950X3D, 16 cores | Table 2, p.11 |
| ZKTorch | GPT-2 (ZKML's distilled model) | 81.3M | – | 10.0 min | 12.00 s | – | Xeon Platinum 8358, 64 threads | Sec 6.3, p.11 |
| ZKTorch | Llama-2-7B (1 token) | 6.74B | 1 | 44.1 min | 100.14 s | 22.9 MB | Xeon Platinum 8358, 64 threads | Table 4, p.12 |
| Jolt Atlas | GPT-2 | 124.0M | – | 38.00 s | – | – | Apple M3 laptop | Table 3, p.19 |
| SLP (5 sampled chunks) | GPT-2 | 124.0M | – | 28.00 s | 160.0 ms | 938 kB | Apple M3 Pro laptop | Sec. 6.2, p.7 |
| LAMP (1 layer, matmuls only) | GPT-2-medium, one layer | 12.6M | 1024 | 3.8 min | 6.52 s | 3.34 MB | AMD EPYC 7B13, 32 cores | Table 5, p.13; Sec 7.4, p.12 |
| Anchuri et al. (1 path) | Llama-2-7B | 6.74B | 64 | 5.8 ms | 12.4 ms | 3.4 MB | RTX 3090 | Sec. 8.2, p.28 |
| Maverick (verif.-only, 1 thr.) | Qwen3-4B | 4.02B | 8 | 354.5 ms | 87.1 ms | 36.1 MB | AWS c8i (client 1 thread) | Table 8, p.16 |
| Maverick (verif.-only, 8 thr.) | Qwen3-4B | 4.02B | 8 | 314.4 ms | 12.8 ms | 36.1 MB | AWS c8i (client 8 threads) | Table 8, p.16 |
| Maverick (standard, 1 thr.) | Qwen3-4B | 4.02B | 8 | 363.3 ms | 89.6 ms | 57.8 MB | AWS c8i (client 1 thread) | Table 8, p.16 |

### Like-for-like ratios against published systems (ours at λ=128 unless noted; their hardware differs)

| published | ours | prover | verifier | proof | their hardware | note |
|---|---|---|---|---|---|---|
| zkCNN: LeNet-5 MNIST | LeNet-5 (MNIST) (C) | 27.1× faster | 4.97× slower | 1.84× larger | AMD EPYC 7R32, 1 core | same model |
| vCNN (re-run by zkCNN): LeNet-5 MNIST | LeNet-5 (MNIST) (C) | 337× faster | 2.91× faster | 385× larger | AMD EPYC 7R32 | same model |
| Bionetta: LeNet-5 | LeNet-5 (MNIST) (C) | 230× faster | 2.88× slower | 149× larger | Xeon E5-2665, 16 threads | same architecture |
| EZKL (run by Bionetta): LeNet-5 | LeNet-5 (MNIST) (C) | 32,867× faster | 69.3× faster | 1.03× larger | Xeon E5-2665, 16 threads | same architecture |
| zkCNN: VGG-11 CIFAR-10 | VGG-11 (CIFAR-10) (C) | 737× faster | 3.41× slower | 7.24× larger | AMD EPYC 7R32, 1 core | same model |
| zkCNN: VGG-16 CIFAR-10 | VGG-16 (CIFAR-10) (C) | 973× faster | 3.27× slower | 10.1× larger | AMD EPYC 7R32, 1 core | same model |
| ZKML: VGG-16 CIFAR-10 | VGG-16 (CIFAR-10) (C) | 7,024× faster | 20.2× slower | 285× larger | AWS r6i.8xlarge, 32 vCPU | same model |
| zkPyTorch: VGG-16 CIFAR-10 | VGG-16 (CIFAR-10) (C) | 24.3× faster | – | – | single CPU core | same model; prover time only |
| ZENO: VGG-16 CIFAR-10 (19.9M-FLOP variant) | VGG-16 (CIFAR-10) (C) | 529× faster | – | – | Xeon Gold 5218 | their variant is smaller |
| ZKML: ResNet-18 CIFAR-10 (ZKML's 281K-param variant) | ResNet-18 (CIFAR-10) (C) | 765× faster | 15.2× slower | 294× larger | AWS r6i.8xlarge, 32 vCPU | their variant has 281K params, ours 11.2M |
| Bionetta: ResNet-18 (Bionetta's version) | ResNet-18 (CIFAR-10) (C) | 204× faster | 12× slower | 5,261× larger | Xeon E5-2665, 16 threads | their own ResNet-18 variant |
| zkGPT: GPT-2 | GPT-2, 64 tok. (C) | 49.7× faster | 2.81× slower | 615× larger | Xeon 6126, 32 threads | their prompt length not in the table |
| zkLLM (re-run by zkGPT): GPT-2 | GPT-2, 64 tok. (C) | 36× faster | 1.82× slower | 493× larger | A100 | their prompt length not in the table |
| DeepProve: GPT-2 | GPT-2, 64 tok. (C) | 77.9× faster | 1.37× faster | 2.87× larger | Ryzen 9 7950X3D, 16 cores | same prompt length |
| DeepProve: GPT-2 | GPT-2, 128 tok. (C) | 106× faster | 1.33× faster | 3.66× larger | Ryzen 9 7950X3D, 16 cores | same prompt length |
| DeepProve: GPT-2 | GPT-2, 256 tok. (C) | 179× faster | 1.22× faster | 5.26× larger | Ryzen 9 7950X3D, 16 cores | same prompt length |
| DeepProve: GPT-2 | GPT-2, 512 tok. (C) | 277× faster | 1.29× slower | 8.37× larger | Ryzen 9 7950X3D, 16 cores | same prompt length |
| zkLLM: OPT-125M | OPT-125M, 2048 tok. (C) | 49.4× faster | 64.7× slower | 5,195× larger | A100 40GB | same prompt length |
| zkLLM: OPT-1.3B | OPT-1.3B, 2048 tok. (C) | 17× faster | 141× slower | 25,898× larger | A100 40GB | same prompt length |
| zkLLM: OPT-6.7B | OPT-6.7B, 2048 tok. (C) | 11× faster | 97.5× slower | 64,338× larger | A100 40GB | same prompt length |
| zkLLM: Llama-2-7B | Llama-2-7B, 2048 tok. (C) | 19.1× faster | 93.6× slower | 63,313× larger | A100 40GB | same prompt length |
| zkLLM (re-run by Anchuri): Llama-2-7B | Llama-2-7B, 2048 tok. (C) | 12× faster | – | – | RTX 3090 | their prompt length not stated |
| ZKTorch: Llama-2-7B (1 token) | Llama-2-7B, 64 tok. (C) | 229× faster | 14× faster | 33.3× larger | Xeon Platinum 8358, 64 threads | they prove 1 token, we prove 64 |
| Maverick (verif.-only, 1 thr.): Qwen3-4B | Qwen3-4B, 8 tok. (Kpre) | 1.41× slower | 15.7× slower | equal | AWS c8i (client 1 thread) | λ=40 and 1 client thread, as Maverick |

### What each system's error bound means

| system | setting | error type | stated error | assumes |
|---|---|---|---|---|
| Anchuri et al. | C | empirical | trace separation (no negligible bound); single-node tamper detected w.p. 1/N per path | separation assumption; CM honest |
| Ours (whole-network check) | C / K | statistical | <= sum_l [p^-r + ((k-1)/n)^t], set to 2^-lambda | CM honest (as in Anchuri); interactive, or Fiat-Shamir with +64 bits |
| SafetyNets | K | statistical | 3b sum n_i / p < 2^-30 | quadratic activations only |
| Slalom | K | statistical | 40 bits per layer (p = 2^24-3, k = 2) | SGX enclave trusted |
| Maverick | K | statistical | kappa = 40 bits | client knows weights (preprocessing) |
| LAMP | C | computational | (k-1)/|F| + 4(1-delta)^t + ..., t=128, BN254 | commit-and-prove |
| zkCNN | C | computational | O(d log|C|/|F|), BLS12-381 ('128-bit') | discrete log, ROM |
| zkLLM | C | computational | negligible, |F| ~ 2^254 (BLS12-381) | discrete log, ROM |
| zkGPT | C | computational | ~100 bits (BN254) | discrete log, ROM |
| DeepProve | C | computational | round-by-round sound; not quantified | ROM |
| ZKML | C | computational | inherits halo2; not quantified | KZG setup or IPA |
| ZKTorch | C | computational | union bound, 'negligible' (BN254) | AGM, KZG setup |
| EZKL | C | computational | inherits halo2; not stated | KZG setup |
| Bionetta | C | computational | GGM + ROM; not quantified | trusted setup per model |
| ZEN / ZENO / vCNN | C | computational | knowledge soundness (GGM/KEA); not quantified | trusted setup |
| Mystique | C | computational+statistical | lambda = 128, rho >= 40 | designated verifier, interactive |
| NanoZK | C | computational | 3(L+2) 2^-128 | IPA, SHA-256 chain |
| TOPLOC | K | empirical | FPR/FNR 0% in evaluation | verifier re-runs the model |
| SVIP | K | empirical | FNR < 5%, FPR < 3% per query | secret proxy task |
| SLP | C | computational+sampling | per audit <= q(M,b,s) + eps; coverage 3/161 on 70B | BN254 HyperKZG |
| Verde / opML / TAO | C | trust | 1-of-k honest (Verde/opML); empirical thresholds (TAO) | an honest provider or validator |
