# LightRFT

<div align="center">

<img src="assets/logo.png" alt="LightRFT Logo" width="600"/>

**Light, Efficient, Omni-modal & Reward-model Driven Reinforcement Fine-Tuning Framework**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/DeepLink-org/lightrft)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

English | [简体中文](README_zh.md)

</div>

---

## 📢 Project Overview

LightRFT is a lightweight post-training framework for LLM models, designed to support post-training exploration in the security and multimodal domains.

## 📖 Introduction

**LightRFT** (Light Reinforcement Fine-Tuning) is an advanced reinforcement learning fine-tuning framework designed for Large Language Models (LLMs) and Vision-Language Models (VLMs). This framework provides efficient and scalable RLHF (Reinforcement Learning from Human Feedback) and RLVR training capabilities, supporting multiple state-of-the-art algorithms and distributed training strategies.

### ✨ Key Features

- 🚀 **High-Performance Inference Engines**
  - Integrated vLLM and SGLang for efficient sampling and inference
  - FP8 inference optimization for significantly reduced latency and memory usage
  - Flexible engine sleep/wake mechanisms for optimal resource utilization

- 🧠 **Rich Algorithm Ecosystem** 
  - **Policy Optimization**: GRPO, GSPO, GMPO, Dr.GRPO
  - **Advantage Estimation**: REINFORCE++, CPGD
  - **Reward Processing**: Reward Norm/Clip
  - **Sampling Strategy**: FIRE Sampling, Token-Level Policy
  - **Stability Enhancement**: DAPO, select_high_entropy_tokens

- 🔧 **Flexible Training Strategies**
  - FSDP (Fully Sharded Data Parallel) v2 support
  - DeepSpeed ZeRO (Stage 1/2/3) support
  - Gradient checkpointing and mixed precision training (BF16/FP16)
  - Adam Offload and memory optimization techniques
  - Support packing samples

- 🎯 **Innovative Resource Collaboration**
  - **Colocate Anything**: Co-locate reward models with training models to maximize GPU utilization
    - Support multiple reward models for parallel inference on the same device
    - Dynamic memory management with automatic training/inference phase switching
    - Reduced cross-device communication overhead for improved end-to-end training efficiency
  - **Balance Anything** 🚧 (Under Development): Intelligent load balancing system
    - Adaptive task scheduling and resource allocation
    - Automatic load balancing for multi-node training
    - Performance optimization for heterogeneous hardware environments

- 🌐 **Comprehensive Multimodal Support**
  - **Native Vision-Language Model (VLM) Training**
    - Support for mainstream VLMs like Qwen-VL, InternVL
    - Parallel processing of multimodal image-text data
    - Support sequence parallel
    - Efficient multimodal tokenization and batching
    - Support processing one text with more than one images
  - **Multimodal Reward Modeling**
    - Support for multiple visual reward models working in collaboration
    - Joint optimization of image understanding and text generation
    - Support reward model as a service
  - **Complete Vision-Language Alignment Training Pipeline**
    - Optimized for multimodal RLVR/RLHF training
    - Built-in support for vision-language model fine-tuning

- 📊 **Complete Experimental Toolkit**
  - Weights & Biases (W&B) integration
  - Math capability benchmarking (GSM8K, Geo3K, etc.)
  - Trajectory saving and analysis tools
  - Automatic checkpoint management

- ✈️ **Efficient Performance Optimization Strategies**
  - Support data load balancing during mixed text-image training, reducing training time by 30%
  - Support dynamic batch size
  - Support memory optimization for logprobs calculation

---

## 🎯 Supported Algorithms

For detailed algorithm descriptions, implementation details, and usage guide, see [Algorithm Documentation](docs/source/quick_start/algorithms.md).

| Algorithm | Type | Key Improvement | Paper |
|-----------|------|-----------------|-------|
| **GRPO** | Policy Optimization | Group normalized advantage estimation |  [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300)  |
| **GSPO** | Policy Optimization | Group sequence policy optimization | [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| **GMPO (WIP)** | Policy Optimization | Geometric-mean policy optimization | [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) |
| **Dr.GRPO** | Policy Optimization | Length bias mitigation | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **DAPO** | Policy Optimization | Decoupled clip and dynamic sampling policy optimization | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **REINFORCE++** | Advantage Estimation | Improved baseline estimation | [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) |
| **CPGD** | Advantage Estimation | KL-based drift constraint | [arXiv:2505.12504](https://arxiv.org/abs/2505.12504) |
| **FIRE Sampling** | Sampling Strategy | Filtering and ranking strategies | [arXiv:2410.21236](https://arxiv.org/abs/2410.21236) |

---

## 🚀 Quick Start

### Requirements

- Python >= 3.10
- CUDA >= 12.8
- PyTorch >= 2.5.1

### Docker Images

TO BE DONE

### Installation

Clone and install LightRFT:

```bash
# Clone the repository
git clone https://github.com/DeepLink-org/LightRFT.git
cd LightRFT

# Install dependencies
pip install -r requirements.txt

# Install LightRFT
pip install -e .
```


## 📚 Usage Guide

### Basic Example: GRPO Training

```bash
# Single node, 8 GPU training example
cd LightRFT

# Run GRPO training (GSM8K math reasoning task)
bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

# Or run Geo3K geometry problem training (VLM multimodal)
bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

---

## 🏗️ Project Structure

```
LightRFT/
├── lightrft/                      # Core library
│   ├── strategy/                  # Training & inference strategies
│   │   ├── fsdp/                  # FSDP implementation
│   │   ├── deepspeed/             # DeepSpeed implementation
│   │   ├── vllm_utils/            # vLLM utilities
│   │   ├── sglang_utils/          # SGLang utilities
│   │   └── utils/                 # Strategy utilities
│   ├── models/                    # Model definitions
│   │   ├── actor_al.py            # Audio-language model actor
│   │   ├── actor_language.py      # Language model actor
│   │   ├── actor_vl.py            # Vision-language model actor
│   │   ├── grm_vl.py              # Generative reward model (Vision-Language)
│   │   ├── srm_al.py              # Scalar reward model (Audio-Language)
│   │   ├── srm_vl.py              # Scalar reward model (Vision-Language)
│   │   ├── loss.py                # Loss functions
│   │   ├── monkey_patch/          # Model adaptation patches for distributed training
│   │   ├── tests/                 # Model tests
│   │   └── utils.py               # Model utilities
│   ├── trainer/                   # Trainer implementations
│   │   ├── ppo_trainer.py         # LLM PPO trainer
│   │   ├── ppo_trainer_vl.py      # VLM PPO trainer
│   │   ├── spmd_ppo_trainer.py    # SPMD PPO trainer Extension (Core)
│   │   ├── grm_trainer_vl.py      # Generative reward model trainer (Vision-Language)
│   │   ├── srm_trainer_al.py      # Scalar reward model trainer (Audio-Language)
│   │   ├── srm_trainer_vl.py      # Scalar reward model trainer (Vision-Language)
│   │   ├── fast_exp_maker.py      # Fast experience generator (Core)
│   │   ├── experience_maker.py    # Base experience generator
│   │   ├── experience_maker_vl.py # Base experience generator for VLM
│   │   ├── replay_buffer.py       # Replay buffer
│   │   ├── replay_buffer_vl.py    # VLM replay buffer
│   │   ├── replay_buffer_utils.py # Replay buffer utilities
│   │   ├── kl_controller.py       # KL divergence controller
│   │   └── utils.py               # Trainer utilities
│   ├── datasets/                  # Dataset processing
│   │   ├── audio_alpaca.py        # Audio Alpaca dataset
│   │   ├── grm_dataset.py         # Generative reward model dataset
│   │   ├── hpdv3.py               # HPDv3 reward model dataset
│   │   ├── image_reward_db.py     # Image reward database
│   │   ├── imagegen_cot_reward.py # Image generation CoT generative reward
│   │   ├── omnirewardbench.py     # OmniRewardBench dataset
│   │   ├── process_reward_dataset.py # Reward dataset processing
│   │   ├── prompts_dataset.py     # LLM Prompts dataset
│   │   ├── prompts_dataset_vl.py  # Vision-language prompts dataset
│   │   ├── rapidata.py            # Rapidata reward modeldataset
│   │   ├── sft_dataset.py         # SFT dataset
│   │   ├── sft_dataset_vl.py      # VLM SFT dataset
│   │   ├── srm_dataset.py         # Scalar reward model base dataset
│   │   └── utils.py               # Dataset utilities
│   └── utils/                     # Utility functions
│       ├── ckpt_scripts/          # Checkpoint processing scripts
│       ├── cli_args.py            # CLI argument parsing
│       ├── distributed_sampler.py # Distributed sampler
│       ├── logging_utils.py       # Logging utilities
│       ├── processor.py           # Data processor for HF model
│       ├── remote_rm_utils.py     # Remote reward model utilities
│       ├── timer.py               # Timer utilities
│       ├── trajectory_saver.py    # Trajectory saver
│       └── utils.py               # General utilities
│
├── examples/                      # Usage examples
│   ├── gsm8k_geo3k/               # GSM8K/Geo3K math reasoning training examples
│   ├── grm_training/              # Generative reward model training examples
│   ├── srm_training/              # Scalar reward model training examples
│   ├── chat/                      # Model dialogue examples
│
├── docs/                          # 📚 Sphinx documentation
│   ├── Makefile                   # Documentation build Makefile
│   ├── make.bat                   # Documentation build batch file
│   └── source/                    # Documentation source
│       ├── _static/               # Static files (CSS, etc.)
│       ├── api_doc/               # API documentation
│       ├── best_practice/         # Best practices & resources
│       ├── installation/          # Installation guides
│       └── quick_start/           # Quick start & user guides
│
├── assets/                        # Assets
│   └── logo.png                   # Project logo
│
├── CHANGELOG.md                   # Changelog
├── LICENSE                        # License file
├── Makefile                       # Project Makefile
├── README.md                      # Project documentation (English)
├── README_zh.md                   # Project documentation (Chinese)
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies
├── requirements-doc.txt           # Documentation dependencies
└── setup.py                       # Package setup script
```

### 🔑 Key Directory Descriptions

- **`lightrft/`**: LightRFT core library, providing training strategies, model definitions, and trainer implementations
- **`examples/`**: Complete training examples and scripts
  - `gsm8k_geo3k/`: GSM8K and Geo3K math reasoning training examples
  - `grm_training/`: Generative reward model training examples
  - `srm_training/`: Scalar reward model training examples
  - `chat/`: Model dialogue examples
- **`docs/`**: Sphinx documentation with complete user guides and API documentation

---

## ⚙️ Key Configuration Parameters

### Batch Size Configuration

```bash
TBS=128                           # Training batch size
RBS=128                            # Rollout batch size
micro_train_batch_size=1          # Micro batch size per GPU
micro_rollout_batch_size=2        # Rollout micro batch size
```

### Algorithm Parameters

```bash
--advantage_estimator group_norm  # Advantage estimator: group_norm, reinforce, cpgd
--n_samples_per_prompt 8          # Number of samples per prompt
--max_epochs 1                    # Training epochs per episode
--num_episodes 3                  # Total training episodes
--kl_estimator k3                 # KL estimator type
--init_kl_coef 0.001              # KL penalty coefficient
```

### Distributed Training

```bash
--fsdp                            # Enable FSDP
--zero_stage 3                    # DeepSpeed ZeRO Stage
--gradient_checkpointing          # Gradient checkpointing
--adam_offload                    # Adam optimizer offload
--bf16                            # BF16 mixed precision
```

### Inference Engine

```bash
--rm_use_engine                   # Use inference engine (vLLM/SGLang) for reward model
--engine_mem_util 0.4             # Engine memory utilization
--engine_tp_size 1                # Engine tensor parallelism degree
--enable_engine_sleep             # Enable engine sleep mechanism
```

---

## ⚡ Performance Optimization Recommendations

### Memory Optimization

1. Use gradient checkpoint

```bash
--gradient_checkpointing
```

2. Use Adam Offload

```bash
--adam_offload
```

3. adjust engine memory utilization

```bash
--engine_mem_util 0.4
```

4. environment variable optimization
```bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
```

### Calculation optimization

1. FP8 rollout (only in the inference stage)
  - Reduce inference latency and VRAM usage, while maintaining BF16 precision during training.

```bash
--fp8_rollout
--enable_vllm_is_correction
```

2. Flash Attention

```bash
--flash_attn
```

3. batch size optimize

Recommend: train_batch_size >= rollout_batch_size × n_samples_per_prompt

---

## 🔧 Troubleshooting

See training scripts for detailed parameter validation logic.

### 1. OOM (Out of Memory)

**Solutions**:
- Reduce `micro_train_batch_size` and `micro_rollout_batch_size`
- Enable `--gradient_checkpointing`
- Lower `--engine_mem_util`
- Use ZeRO Stage 3

### 2. Training Instability

**Solutions**:
- Enable Reward Normalization: `--normalize_reward`
- Lower learning rate
- Use `--advantage_estimator group_norm`
- Try DAPO algorithm

---

## 📖 Documentation

### 📚 Complete Documentation Guide

**Quick Start:**
- [Installation Guide](docs/source/installation/index.rst) - Docker images, installation methods, and troubleshooting
- [Supported Algorithms](docs/source/quick_start/algorithms.md) - Comprehensive algorithm guide with implementation details
- [Configuration Reference](docs/source/quick_start/configuration.md) - Complete parameter documentation

**Best Practices:**
- [Training Strategy Usage](docs/source/best_practice/strategy.rst) - FSDP, DeepSpeed, and inference engine configuration
- [FAQ](docs/source/best_practice/faq.md) - Frequently asked questions and solutions
- [Troubleshooting Guide](docs/source/best_practice/troubleshooting.md) - Common issues and debugging
- [Contributing Guide](docs/source/best_practice/contributing.md) - How to contribute to LightRFT

### Build Documentation Locally

Install documentation dependencies:
```bash
pip install -r requirements-doc.txt
```

Generate HTML documentation:
```bash
make docs
# Open docs/build/index.html to view documentation
```

Live documentation preview:
```bash
make docs-live
# Visit http://localhost:8000
```

## 🤝 Contributing

We welcome community contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Code formatting (YAPF)
make format

# Code linting (Flake8)
make fcheck
```

---

## 📚 Citation

If you use this codebase in your research or applications, please cite it as follows:

```bibtex
@misc{lightrft,
  title={LightRFT},
  author={Niu, Yazhe and Pu, Yuan and Shi, Dongxing and Lu, Yudong and Xiong, Yingtong and Ge, Ruijun and Sun, Jiaxuan and Wan, Zunian and Zhang, Shaoang and others},
  publisher={GitHub},
  howpublished={\url{https://github.com/DeepLink-org/LightRFT}},
  year={2025},
}
```

## ⌨️ Development Team

- Business Team Framework Group: Responsible for the development of the algorithm ecosystem, training strategies, multimodal support, and the experimental toolchain, focusing on algorithm innovation and the enhancement of model training capabilities.
- System Team DeepLink: Responsible for high-performance inference engines, resource coordination mechanisms, system-level performance optimization, and underlying infrastructure, focusing on the optimization of system performance and resource utilization efficiency.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**LightRFT**​ is collaboratively developed by **The RL Team​ of the Safe and Trustworthy Center** and **The System Platform Center DeepLink Team**​ at Shanghai AI Laboratory. We extend our sincere thanks to the contributors from both teams.
- The reinforcement learning component of this project is developed based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), and we express our heartfelt gratitude to its development team for their outstanding work.
- We thank the Qwen, InternVL, and DeepSeek teams for providing excellent open-source foundation models.
- We also acknowledge the powerful tools provided by open-source communities such as DeepSpeed, PyTorch, vLLM, and SGLang.

### Open Source Dependencies

This project builds upon the following outstanding open-source projects (including but not limited):

- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**, **[verl](https://github.com/volcengine/verl)** - Core RL framework foundation (parts of key components adapted and reused)
- [vLLM](https://github.com/vllm-project/vllm) - High-performance inference engine
- [SGLang](https://github.com/sgl-project/sglang) - Structured generation language runtime
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Distributed training optimization
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) - Fully Sharded Data Parallel

Thanks to all contributors and supporters!

---

## 🗓️ RoadMap

We are actively working on the following improvements and features:

### Core Feature Enhancements

- [ ] **Trajectory Functionality Extension**
  - Add more analysis metrics
  - Enhanced trajectory saving and analysis capabilities

- [ ] **Reward Mechanism Refactoring**
  - Refactor rule-based and model-based reward computation
  - Optimize reward dataset processing pipeline

### Algorithm Optimization & Integration

- [ ] **More Algorithm Integration**
  - Entropy-based token selection 
  - GMPO (Geometric-Mean Policy Optimization)
  - GSPO (Group Sequence Policy Optimization)

- [ ] **Advantage Computation Refactoring**
  - Optimize advantage estimation module architecture
  - Unify advantage computation interface across algorithms

- [ ] **Loss-Filter Mechanism Optimization**
  - Refactor loss filtering implementation
  - Complete GSM8K/Geo3K benchmark experiments
  - Document experimental results and analysis



Community contributions and feedback are welcome!

---

## 📮 Contact

For questions or suggestions, please contact us via:

- **Issues**: [GitHub Issues](https://github.com/DeepLink-org/LightRFT/issues)
- **Email**: DeepLink-org@pjlab.org.cn


---

<div align="center">

**⭐ If this project helps you, please give us a star!**

Made with ❤️ by LightRFT Team

</div>
