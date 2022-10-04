import numpy as np
import torch
from torch.functional import Tensor

from synth.nn.abstractions import ucfg_bigram
from synth.nn.u_grammar_predictor import UGrammarPredictorLayer
from synth.syntax.grammars.u_cfg import UCFG
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
cfg = UCFG.depth_constraint(dsl, FunctionType(INT, INT), 4)
cfg2 = UCFG.depth_constraint(dsl, FunctionType(FunctionType(INT, INT), INT, INT), 5)


def test_forward() -> None:
    layer = UGrammarPredictorLayer(50, {cfg}, ucfg_bigram)
    generator = torch.manual_seed(0)
    for _ in range(20):
        x = torch.randn((100, 50), generator=generator)
        y: Tensor = layer(x)
        assert y.shape == torch.Size([x.shape[0], 16])


def test_to_logpcfg() -> None:
    layer = UGrammarPredictorLayer(50, {cfg}, ucfg_bigram)
    generator = torch.manual_seed(0)
    for _ in range(20):
        x = torch.randn((5, 50), generator=generator)
        y = layer(x)
        for i in range(y.shape[0]):
            log_pcfg = layer.tensor2log_prob_grammar(
                y[i], cfg.type_request, total_variable_order=False
            )
            for S in log_pcfg.rules:
                total = sum(
                    sum(np.exp(t_prob.item()) for t_prob in dico.values())
                    for P, dico in log_pcfg.tags[S].items()
                )
                assert np.isclose(1, total)


def test_logpcfg2pcfg() -> None:
    layer = UGrammarPredictorLayer(50, {cfg}, ucfg_bigram)
    generator = torch.manual_seed(0)
    for _ in range(20):
        x = torch.randn((5, 50), generator=generator)
        y = layer(x)
        for i in range(y.shape[0]):
            log_pcfg = layer.tensor2log_prob_grammar(
                y[i], cfg.type_request, total_variable_order=False
            )
            pcfg = log_pcfg.to_prob_u_grammar()
            pcfg.init_sampling(0)
            P = pcfg.sample_program()
            # This line is important because it checks that a call to log_probability does not affect the probabilities
            log_pcfg.log_probability(P).item()
            for S in pcfg.rules:
                for P in pcfg.rules[S]:
                    for v in pcfg.rules[S][P]:
                        target = np.exp(log_pcfg.tags[S][P][tuple(v)].item())
                        assert np.isclose(
                            pcfg.probabilities[S][P][tuple(v)], target
                        ), f"S:{S}, P:{P} V:{v} pcfg_prob:{pcfg.probabilities[S][P][tuple(v)]} log_pcfg_prob:{target}"


def test_var_as_function() -> None:
    layer = UGrammarPredictorLayer(50, {cfg2, cfg}, ucfg_bigram)
    generator = torch.manual_seed(0)
    for _ in range(5):
        for c in [cfg, cfg2]:
            x = torch.randn((5, 50), generator=generator)
            y = layer(x)
            for i in range(y.shape[0]):
                log_pcfg = layer.tensor2log_prob_grammar(
                    y[i], c.type_request, total_variable_order=False
                )
                pcfg = log_pcfg.to_prob_u_grammar()
                pcfg.init_sampling(0)
                P = pcfg.sample_program()
                prob = pcfg.probability(P)
                exp_logprob = np.exp(log_pcfg.log_probability(P).item())
                assert np.isclose(
                    prob, exp_logprob
                ), f"P:{P} prob:{prob} explogprob:{exp_logprob}"


def test_varprob() -> None:
    layer = UGrammarPredictorLayer(10, {cfg}, ucfg_bigram)
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
            layer.tensor2log_prob_grammar(
                y[i], cfg.type_request, total_variable_order=False
            )
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
                        prob = np.exp(pcfg.tags[S][P][()].item())
                        assert np.isclose(
                            prob, layer.variable_probability
                        ), f"S:{S}, P:{P} prob:{prob} target:{layer.variable_probability}"


def test_learning() -> None:
    layer = UGrammarPredictorLayer(10, {cfg}, ucfg_bigram)
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
        pcfgs = [
            layer.tensor2log_prob_grammar(y[i], cfg.type_request)
            for i in range(batch_size)
        ]
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


def test_learning_cross_entropy() -> None:
    layer = UGrammarPredictorLayer(10, {cfg}, ucfg_bigram)
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
        opti.zero_grad()
        loss = layer.loss_cross_entropy(
            programs, [cfg.type_request for _ in programs], y
        )
        loss.backward()
        opti.step()

        with torch.no_grad():
            pcfgs = [
                layer.tensor2log_prob_grammar(y[i], cfg.type_request)
                for i in range(batch_size)
            ]
            logprob = -layer.loss_negative_log_prob(
                programs, pcfgs, length_normed=False
            ).item()
            mean_prob.append(np.exp(logprob))

    for i in range(1, len(mean_prob)):
        assert mean_prob[i - 1] < mean_prob[i], f"{mean_prob}"

    assert mean_prob[-1] > 0.12
