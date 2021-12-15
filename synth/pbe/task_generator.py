from typing import Callable, Dict, Generator, List as TList, Any

from synth.task import Task
from synth.specification import PBE, Example
from synth.semantic.evaluator import Evaluator
from synth.syntax.type_system import Arrow, Type
from synth.syntax.program import Program
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.generation.sampler import Sampler


class TaskGenerator:
    def __init__(
        self,
        input_generator: Sampler,
        evaluator: Evaluator,
        gen_random_type_request: Callable[[], Type],
        gen_random_sample_number: Callable[[Type], int],
        type2pcfg: Dict[Type, ConcretePCFG],
        output_validator: Callable[[Any], bool],
        max_tries: int = 100,
    ) -> None:
        self.input_generator = input_generator
        self.evaluator = evaluator
        self.gen_random_type_request = gen_random_type_request
        self.gen_random_sample_number = gen_random_sample_number
        self.type2pcfg = type2pcfg
        self.max_tries = max_tries
        self.output_validator = output_validator
        # For statistics
        self.difficulty: Dict[Type, TList[int]] = {}

    def __generate_program__(self, type_request: Type) -> Program:
        nargs: int = (
            0 if not isinstance(type_request, Arrow) else len(type_request.arguments())
        )
        solution: Program = self.type2pcfg[type_request].sample_program()
        while not solution.is_using_all_variables(nargs):
            solution = self.type2pcfg[type_request].sample_program()
        return solution

    def generate_task(self) -> Task[PBE]:
        type_request = self.gen_random_type_request()
        arguments = (
            [] if not isinstance(type_request, Arrow) else type_request.arguments()
        )
        if type_request not in self.difficulty:
            self.difficulty[type_request] = [0, 0]

        # Generate correct program that makes use of all variables
        solution = self.__generate_program__(type_request)
        # Try to generate the required number of samples
        samples = self.gen_random_sample_number(type_request)
        inputs: TList = []
        outputs = []
        tries = 0
        # has_enough_tries_to_reach_desired_no_of_samples and has_remaining_tries
        while (self.max_tries - tries) + len(
            inputs
        ) >= samples and tries < self.max_tries:
            tries += 1
            new_input = [
                self.input_generator.sample(arg_type) for arg_type in arguments
            ]
            try:
                output = self.evaluator.eval(solution, new_input)
            except:
                # Catch errors just in case
                continue
            if self.output_validator(output):
                inputs.append(new_input)
                outputs.append(output)
                if len(inputs) >= samples:
                    break

        self.difficulty[type_request][0] += tries
        self.difficulty[type_request][1] += tries - len(inputs)

        # Sample another task if failed
        if len(inputs) < samples:
            return self.generate_task()
        return Task(
            type_request,
            PBE([Example(inp, out) for inp, out in zip(inputs, outputs)]),
            solution,
            {"generated": True, "tries": tries},
        )

    def generator(self) -> Generator[Task[PBE], None, None]:
        while True:
            yield self.generate_task()


def basic_output_validator(
    int_lexicon: TList[int], max_list_length: int
) -> Callable[[Any], bool]:
    def validate_output(output: Any) -> bool:
        if isinstance(output, bool):
            return True
        elif isinstance(output, int):
            return output in int_lexicon
        elif isinstance(output, list):
            return len(output) <= max_list_length and all(
                validate_output(x) for x in output
            )
        return False

    return validate_output
