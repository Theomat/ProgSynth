import numpy as np
import torch
from torch.functional import Tensor

import pytest

from synth.nn.pcfg_predictor import (
    BigramsPredictorLayer,
)
from synth.syntax.grammars.cfg import CFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Function, Primitive, Variable
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
cfg = CFG.from_dsl(dsl, FunctionType(INT, INT), 4)
cfg2 = CFG.from_dsl(dsl, FunctionType(FunctionType(INT, INT), INT, INT), 5)


layers = [BigramsPredictorLayer]


def test_forward() -> None:
    layer = BigramsPredictorLayer(50, {cfg})
    generator = torch.manual_seed(0)
    nb_parents = len(syntax) + 1 - 1
    max_args = 2
    for _ in range(20):
        x = torch.randn((100, 50), generator=generator)
        y: Tensor = layer(x)
        assert y.shape == torch.Size([x.shape[0], 15])


@pytest.mark.parametrize("layer_class", layers)
def test_to_logpcfg(layer_class: BigramsPredictorLayer) -> None:
    layer = layer_class(50, {cfg})
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


@pytest.mark.parametrize("layer_class", layers)
def test_logpcfg2pcfg(layer_class: BigramsPredictorLayer) -> None:
    layer = layer_class(50, {cfg})
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
            # This line is important because it checks that a call to log_probability does not affect the probabilities
            log_pcfg.log_probability(P).item()
            for S in pcfg.rules:
                for P in pcfg.rules[S]:
                    target = np.exp(log_pcfg.rules[S][P][1].item())
                    assert np.isclose(
                        pcfg.rules[S][P][1], target
                    ), f"S:{S}, P:{P} pcfg_prob:{pcfg.rules[S][P][1]} log_pcfg_prob:{target}"


@pytest.mark.parametrize("layer_class", layers)
def test_var_as_function(layer_class: BigramsPredictorLayer) -> None:
    layer = layer_class(50, {cfg2, cfg})
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


@pytest.mark.parametrize("layer_class", layers)
def test_varprob(layer_class: BigramsPredictorLayer) -> None:
    layer = layer_class(10, {cfg})
    opti = torch.optim.AdamW(layer.parameters(), lr=1e-1)
    steps = 10
    batch_size = 10
    programs = [
        Function(
            Primitive("+", FunctionType(INT, INT, INT)),
            [Variable(0, INT), Primitive("1", INT)],
        )
    ] * batch_size
    for _ in range(steps):
        inputs = torch.ones((batch_size, 10))
        y = layer(inputs)
        pcfgs = [
            layer.tensor2pcfg(y[i], cfg.type_request, total_variable_order=False)
            for i in range(batch_size)
        ]
        opti.zero_grad()
        loss = layer.loss_negative_log_prob(programs, pcfgs)
        loss.backward()
        opti.step()

        for pcfg in pcfgs:
            for S in pcfg.rules:
                for P in pcfg.rules[S]:
                    if isinstance(P, Variable):
                        prob = np.exp(pcfg.rules[S][P][1].item())
                        assert np.isclose(
                            prob, layer.variable_probability
                        ), f"S:{S}, P:{P} prob:{prob} target:{layer.variable_probability}"


def test_learning() -> None:
    layer = BigramsPredictorLayer(10, {cfg})
    opti = torch.optim.AdamW(layer.parameters(), lr=1e-1)
    steps = 10
    mean_prob = []
    batch_size = 10
    programs = [
        Function(
            Primitive("+", FunctionType(INT, INT, INT)),
            [Variable(0, INT), Primitive("1", INT)],
        )
    ] * batch_size
    for step in range(steps):
        inputs = torch.ones((batch_size, 10))
        y = layer(inputs)
        pcfgs = [layer.tensor2pcfg(y[i], cfg.type_request) for i in range(batch_size)]
        opti.zero_grad()
        loss = layer.loss_negative_log_prob(programs, pcfgs)
        loss.backward()
        opti.step()

        with torch.no_grad():
            logprob = -layer.loss_negative_log_prob(
                programs, pcfgs, length_normed=False
            ).item()
            mean_prob.append(np.exp(logprob))

    for i in range(1, len(mean_prob)):
        assert mean_prob[i - 1] < mean_prob[i], f"{mean_prob}"

    assert mean_prob[-1] > 0.12
