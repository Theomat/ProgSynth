from random import randint
from typing import (
    List as TList,
    Any,
    Optional,
    Tuple,
)

from synth.pbe.task_generator import (
    TaskGenerator,
    basic_output_validator,
    reproduce_dataset,
)
from synth.syntax.program import Program
from synth.syntax.type_system import Type

from synth.task import Dataset, Task
from synth.specification import PBE, Example
from synth.semantic.evaluator import Evaluator
from synth.syntax import (
    BOOL,
    STRING,
    PrimitiveType,
    DSL,
)
from synth.generation import (
    LexiconSampler,
    UnionSampler,
)


N = PrimitiveType("N")
U = PrimitiveType("U")
L = PrimitiveType("L")
O = PrimitiveType("O")
primitives = {"N": N, "U": U, "L": L, "O": O}
repeated_primitives = {"*": (0, 3), "+": (1, 3), "?": (0, 1)}


class RegexpTaskGenerator(TaskGenerator):
    def __generate_program__(self, type_request: Type) -> Tuple[Program, bool]:
        """
        (program, is_unique)
        """
        program, is_unique = super().__generate_program__(type_request)
        self.__successful_tries = 0
        self.__regexp = "".join(program.__str__().split("(")[2:]).split(" ")
        return program, is_unique

    def __sample_input__(self, arguments: TList[Type]) -> TList:
        new_input = [""]
        # true examples
        if self.__successful_tries < 3:
            repeated = None
            for r in self.__regexp:
                times = 1
                if repeated:
                    left, right = repeated_primitives[repeated]
                    times = randint(left, right)
                    repeated = None
                if r in primitives.keys():
                    [
                        new_input.insert(
                            0, self.input_generator.sample(type=primitives[r])
                        )
                        for _ in range(0, times)
                    ]
                elif r == "W":
                    [new_input.insert(0, " ") for _ in range(0, times)]
                elif r in repeated_primitives.keys():
                    repeated = r
            new_input = [new_input]
        else:
            new_input = [
                self.input_generator.sample(type=arg_type) for arg_type in arguments
            ]
        return new_input

    def __eval_input__(self, solution: Program, new_input: TList) -> Any:
        try:
            output = self.evaluator.eval(solution, new_input)
            if self.__successful_tries < 3 and output:
                self.__successful_tries += 1
            return output
        except Exception as e:
            if type(e) in self.skip_exceptions:
                return None
            else:
                raise e


def reproduce_regexp_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    *args: Any,
    **kwargs: Any
) -> Tuple[RegexpTaskGenerator, TList[int]]:

    str_lexicon = list([chr(i) for i in range(32, 126)])
    n_lexicon = [chr(i) for i in range(48, 58)]
    u_lexicon = [chr(i) for i in range(65, 91)]
    l_lexicon = [chr(i) for i in range(97, 123)]
    o_lexicon = str_lexicon[1:]  # without whitespace
    o_lexicon = list(set(o_lexicon) - set(n_lexicon + u_lexicon + l_lexicon))
    input_sampler = UnionSampler(
        {
            O: LexiconSampler(o_lexicon, seed=seed),
            N: LexiconSampler(n_lexicon, seed=seed),
            U: LexiconSampler(u_lexicon, seed=seed),
            L: LexiconSampler(l_lexicon, seed=seed),
            STRING: LexiconSampler(str_lexicon, seed=seed),
            BOOL: LexiconSampler([True, False], seed=seed),
        }
    )
    task_generator, str_lexicon = reproduce_dataset(
        dataset,
        dsl,
        evaluator,
        None,
        lambda _, __: None,
        lambda _: input_sampler,
        lambda _, max_list_length: basic_output_validator(
            {str: str_lexicon, bool: [True, False]}, max_list_length
        ),
        lambda _: str_lexicon,
        seed,
        *args,
        **kwargs
    )

    return (
        RegexpTaskGenerator(
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
