# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum
from functools import lru_cache

import jax


class KernelBackend(Enum):
    jax = "jax"
    pallas = "pallas"


class Accelerator(Enum):
    cpu = "cpu"
    tpu = "tpu"

    @staticmethod
    @lru_cache
    def get_accelerator() -> Accelerator:
        return Accelerator.tpu if jax.default_backend() == "tpu" else Accelerator.cpu

    @staticmethod
    @lru_cache
    def get_kernel_backend() -> KernelBackend:
        return KernelBackend.pallas if Accelerator.get_accelerator() == Accelerator.tpu else KernelBackend.jax

    @staticmethod
    def get_lane_count() -> int:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.tpu:
            lane_count = 128
        else:
            raise ValueError(f"unexpected accelerator ({accelerator})")

        return lane_count
