import torch

import numpy as np
from torch.functional import Tensor

from synth.nn.pcfg_predictor import BigramsPredictorLayer
from synth.syntax.concrete.concrete_cfg import ConcreteCFG
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    FunctionType,
)


syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "1": INT,
}

dsl = DSL(syntax)
cfg = ConcreteCFG.from_dsl(dsl, FunctionType(INT, INT), 4)
cfg2 = ConcreteCFG.from_dsl(dsl, FunctionType(FunctionType(INT, INT), INT, INT), 5)


def test_forward() -> None:
    layer = BigramsPredictorLayer(50, {cfg})
    generator = torch.manual_seed(0)
    nb_parents = len(syntax) + 1 - 1
    max_args = 2
    for _ in range(20):
        x = torch.randn((100, 50), generator=generator)
        y: Tensor = layer(x)
        assert y.shape == torch.Size([x.shape[0], nb_parents, max_args, len(syntax)])
        ones = y.exp().sum(-1)
        assert torch.allclose(ones, torch.ones_like(ones))


def test_to_logpcfg() -> None:
    layer = BigramsPredictorLayer(50, {cfg})
    generator = torch.manual_seed(0)
    for _ in range(20):
        x = torch.randn((5, 50), generator=generator)
        y = layer(x)
        for i in range(y.shape[0]):
            log_pcfg = layer.tensor2pcfg(
                y[i], cfg.type_request, total_variable_order=False
            )
            for S in log_pcfg.rules:
                total = sum(
                    np.exp(log_pcfg.rules[S][P][1].item()) for P in log_pcfg.rules[S]
                )
                assert np.isclose(1, total)


def test_logpcfg2pcfg() -> None:
    layer = BigramsPredictorLayer(50, {cfg})
    generator = torch.manual_seed(0)
    for _ in range(20):
        x = torch.randn((5, 50), generator=generator)
        y = layer(x)
        for i in range(y.shape[0]):
            log_pcfg = layer.tensor2pcfg(
                y[i], cfg.type_request, total_variable_order=False
            )
            pcfg = log_pcfg.to_pcfg()
            pcfg.init_sampling(0)
            P = pcfg.sample_program()
            prob = pcfg.probability(P)
            exp_logprob = np.exp(log_pcfg.log_probability(P).item())
            assert np.isclose(prob, exp_logprob)


def test_var_as_function() -> None:
    layer = BigramsPredictorLayer(50, {cfg2, cfg})
    generator = torch.manual_seed(0)
    for _ in range(5):
        for c in [cfg, cfg2]:
            x = torch.randn((5, 50), generator=generator)
            y = layer(x)
            for i in range(y.shape[0]):
                log_pcfg = layer.tensor2pcfg(
                    y[i], c.type_request, total_variable_order=False
                )
                pcfg = log_pcfg.to_pcfg()
                pcfg.init_sampling(0)
                P = pcfg.sample_program()
                prob = pcfg.probability(P)
                exp_logprob = np.exp(log_pcfg.log_probability(P).item())
                assert np.isclose(prob, exp_logprob)
