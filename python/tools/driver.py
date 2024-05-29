import argparse
from abc import ABC, abstractmethod

import numpy as np
from gc_mlir import ir
from utils import get_default_passes


class DriverParams(ABC):
    @staticmethod
    @abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    def __init__(self, params: dict):
        pass


class Driver(ABC):
    def __init__(self, ctx: ir.Context):
        self.ir_module = self.init_module(ctx)

    @abstractmethod
    def init_module(self, ctx: ir.Context) -> ir.Module:
        pass

    @abstractmethod
    def prepare_np_args(self) -> "list[np.ndarray]":
        pass

    @abstractmethod
    def prepare_np_res(self) -> "list[np.ndarray]":
        pass

    def get_passes(self) -> str:
        return get_default_passes()
