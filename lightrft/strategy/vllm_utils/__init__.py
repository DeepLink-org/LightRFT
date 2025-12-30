# Copyright (c) 2026, LightRFT Team.
"""
This module provides utilities for initializing and configuring a vLLM engine.

The module simplifies the process of creating a vLLM engine with specific configurations
for large language model inference, particularly in reinforcement learning from human feedback
(RLHF) contexts. It offers both high-level and low-level functions for engine creation,
with support for tensor parallelism, memory optimization, and multimodal capabilities.
"""

import logging
from typing import Any

from vllm import LLM

logger = logging.getLogger(__name__)


def get_vllm_engine_for_rollout(args: Any) -> LLM:
    """
    Initialize and configure a vLLM engine for reinforcement learning rollout phase.

    This function creates a vLLM engine instance with configurations provided in the args parameter,
    such as the pretrained model path, tensor parallelism size, and memory utilization settings.
    It handles multimodal configurations automatically based on the provided arguments and serves
    as a high-level wrapper around the base get_vllm_engine function.

    :param args: Configuration arguments for the vLLM engine containing model and runtime parameters.
    :type args: Any

    :return: Configured vLLM engine instance ready for rollout operations.
    :rtype: vllm.LLM

    Example::

        >>> args = argparse.Namespace()
        >>> args.pretrain = "Qwen/Qwen2.5-7B-Instruct"
        >>> args.engine_tp_size = 1
        >>> args.engine_mem_util = 0.6
        >>> args.enable_engine_sleep = True
        >>> args.bf16 = True
        >>> args.prompt_max_len = 2048
        >>> args.generate_max_len = 1024
        >>> args.text_only = False
        >>> args.limit_mm_image_per_prompt = 5
        >>>
        >>> engine = get_vllm_engine_for_rollout(args)

    Note:
        The construction of tensor-parallel (TP) group is implemented in the strategy part.
        Multimodal image limits are automatically configured when applicable.
    """
    if hasattr(args, "limit_mm_image_per_prompt") and not args.text_only:
        kwargs = {"limit_mm_per_prompt": {"image": args.limit_mm_image_per_prompt}}
    else:
        kwargs = {}

    if args.fp8_rollout:
        try:
            import flash_rl
        except ImportError:
            logger.warning(
                "Failed to import flash_rl, fp8 rollout is not supported in this environment. Set fp8_rollout to False"
            )
            args.fp8_rollout = False

    if args.fp8_rollout:
        import vllm.envs as envs
        assert envs.VLLM_USE_V1, 'fp8 rollout only supports vllm v1 for now'
        vllm_engine = LLM(
            model=args.pretrain,
            tensor_parallel_size=args.engine_tp_size,
            gpu_memory_utilization=args.engine_mem_util,
            distributed_executor_backend="external_launcher",
            enable_sleep_mode=args.enable_engine_sleep,
            max_model_len=args.prompt_max_len + args.generate_max_len,
            seed=0,
            quantization='fp8',
            **kwargs,
        )
    else:
        vllm_engine = LLM(
            model=args.pretrain,
            dtype="bfloat16" if args.bf16 else "float16",
            tensor_parallel_size=args.engine_tp_size,
            gpu_memory_utilization=args.engine_mem_util,
            distributed_executor_backend="external_launcher",
            worker_cls="lightrft.strategy.vllm_utils.vllm_worker_wrap_no_ray.WorkerWrap",
            enable_sleep_mode=args.enable_engine_sleep,
            max_model_len=args.prompt_max_len + args.generate_max_len,
            seed=0,
            **kwargs,
        )

    return vllm_engine
