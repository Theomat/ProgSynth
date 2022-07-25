from typing import (
    List as TList,
    Any,
    Optional,
    Tuple,
)

from examples.pbe.regexp.type_regex import REGEXP

from synth.pbe.task_generator import (
    TaskGenerator,
    basic_output_validator,
    reproduce_dataset,
)
from synth.syntax.program import Program
from synth.syntax.type_system import Type

from synth.task import Dataset, Task
from synth.specification import PBE, Example, PBEWithConstants
from synth.semantic.evaluator import Evaluator
from synth.syntax import (
    STRING,
    DSL,
)
from synth.generation import (
    LexiconSampler,
    UnionSampler,
)


class TransductionTaskGenerator(TaskGenerator):
    def __generate_program__(self, type_request: Type) -> Tuple[Program, bool]:
        """
        (program, is_unique)
        """
        program, is_unique = super().__generate_program__(type_request)
        self.__constants = super().__sample_input__([STRING, STRING])
        return program, is_unique

    def __sample_input__(self, arguments: TList[Type]) -> TList:
        arguments = super().__sample_input__(arguments)
        return self.__constants + arguments

    def __make_task__(
        self,
        type_request: Type,
        solution: Program,
        inputs: TList,
        outputs: TList,
        **kwargs: Any
    ) -> Task[PBEWithConstants]:
        return Task(
            type_request,
            PBEWithConstants(
                [Example(inp[2:], out) for inp, out in zip(inputs, outputs)]
            ),
            [self.__constants[0]],
            [self.__constants[1]],
            solution,
            {"generated": True, **kwargs},
        )


def reproduce_transduction_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    *args: Any,
    **kwargs: Any
) -> Tuple[TaskGenerator, TList[int]]:
    def analyser(start: None, elment: Any) -> None:
        pass

    str_lexicon = list([chr(i) for i in range(32, 126)])
    regexp_symbols = [
        "_",
        ")",
        "{",
        "+",
        ";",
        "=",
        "$",
        "\\",
        "^",
        ",",
        "!",
        "*",
        "'",
        " ",
        ">",
        "}",
        "<",
        "[",
        '"',
        "#",
        "|",
        "`",
        "%",
        "?",
        ":",
        "]",
        "&",
        "(",
        "@",
        ".",
        "/",
        "-",
    ]

    def get_sampler(start: None) -> UnionSampler:
        return UnionSampler(
            {
                STRING: LexiconSampler(str_lexicon, seed=seed),
                REGEXP: LexiconSampler(regexp_symbols, seed=seed),
            }
        )

    task_generator, str_lexicon = reproduce_dataset(
        dataset,
        dsl,
        evaluator,
        None,
        lambda _, __: None,
        get_sampler,
        lambda _, max_list_length: basic_output_validator(
            {str: str_lexicon + regexp_symbols}, max_list_length
        ),
        lambda _: str_lexicon + regexp_symbols,
        seed,
        *args,
        **kwargs
    )

    return (
        TransductionTaskGenerator(
            task_generator.input_generator,
            task_generator.evaluator,
            task_generator.gen_random_type_request,
            task_generator.gen_random_sample_number,
            task_generator.type2pgrammar.values(),
            task_generator.output_validator,
            task_generator.max_tries,
            task_generator.uniques,
            task_generator.verbose,
        ),
        str_lexicon,
    )
