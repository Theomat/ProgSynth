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
    def generate_task(self) -> Task[PBE]:
        while True:
            type_request = type_request = self.__generate_type_request__()
            arguments = type_request.arguments()

            # Generate correct program that makes use of all variables
            solution, is_unique = self.__generate_program__(type_request)
            regexp = "".join(solution.__str__().split("(")[2:]).split(" ")
            # Try to generate the required number of samples
            samples = self.gen_random_sample_number.sample(type=type_request)
            inputs: TList = []
            outputs = []
            tries = 0
            successful_tries = 0
            # has_enough_tries_to_reach_desired_no_of_samples and has_remaining_tries
            while (self.max_tries - tries) + len(
                inputs
            ) >= samples and tries < self.max_tries:
                tries += 1
                new_input = [""]
                # true examples
                if successful_tries < 3:
                    repeated = None
                    for r in regexp:
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
                        self.input_generator.sample(type=arg_type)
                        for arg_type in arguments
                    ]
                try:
                    output = self.evaluator.eval(solution, new_input)
                    if successful_tries < 3 and output:
                        successful_tries += 1
                    elif output is None:
                        return self.generate_task()
                except Exception as e:
                    if type(e) in self.skip_exceptions:
                        continue
                    else:
                        raise e
                if self.output_validator(output):
                    inputs.append(new_input)
                    outputs.append(output)
                    if len(inputs) >= samples:
                        break

            self.difficulty[type_request][0] += tries
            self.difficulty[type_request][1] += tries - len(inputs)

            # Sample another task if failed
            if len(inputs) < samples:
                self._failed_types.add(type_request)
                continue
            self._failed_types = set()
            self.generated_types[type_request] += 1
            return Task(
                type_request,
                PBE([Example(inp, out) for inp, out in zip(inputs, outputs)]),
                solution,
                {"generated": True, "tries": tries},
            )


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
        lambda _, max_list_length: basic_output_validator(str_lexicon, max_list_length),
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
