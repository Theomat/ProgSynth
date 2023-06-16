from typing import Optional
from karel_runtime import KarelWorld

import numpy as np


def random_world(
    width: int, height: int, rng: Optional[np.random.Generator] = None
) -> KarelWorld:
    world = KarelWorld(width, height)
    gen = rng or np.random.default_rng()
    world.grid[gen.random((width, height)) > 0.8] = 1
    world.grid[world.karel] = 0
    world.markers = (gen.random((width, height)) > 0.7).astype(int)
    world.markers[world.grid > 0] = 0
    return world


if __name__ == "__main__":
    import argparse
    import tqdm

    from synth import Dataset, PBE, Task, Example
    from synth.utils import chrono
    from synth.syntax import CFG, ProbDetGrammar, auto_type

    from karel import dsl, evaluator

    parser = argparse.ArgumentParser(description="Generate a karel dataset")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="karel.pickle",
        help="output file (default: karel.pickle)",
    )
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed (default: 0)")
    parser.add_argument(
        "-w", "--width", type=int, default=10, help="grid width (default: 10)"
    )
    parser.add_argument(
        "-h", "--height", type=int, default=10, help="grid height (default: 10)"
    )
    parser.add_argument(
        "-g",
        "--grids",
        type=int,
        default=3,
        help="number of grids per task (default: 3)",
    )
    parser.add_argument(
        "--size", type=int, default=100, help="generated dataset size (default: 100)"
    )
    parser.add_argument(
        "--max-operations",
        type=int,
        default=5,
        help="solutions max operations (default: 5)",
    )
    parser.add_argument(
        "--uniform", action="store_true", default=False, help="use uniform PCFGs"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose generation",
    )
    parameters = parser.parse_args()
    output_file: str = parameters.output
    seed: int = parameters.seed
    grids: int = parameters.grids
    width: int = parameters.width
    height: int = parameters.height
    max_depth: int = parameters.max_operations
    gen_dataset_size: int = parameters.size
    uniform: bool = parameters.uniform
    verbose: bool = parameters.verbose
    # ================================
    # Task Generator
    # ================================
    tr = auto_type("world -> result")
    print("Generating dataset...", gen_dataset_size, end="", flush=True)
    with chrono.clock("dataset.generate") as c:
        cfg = CFG.depth_constraint(dsl, tr, max_depth)
        if uniform:
            pcfg = ProbDetGrammar.uniform(cfg)
        else:
            print(
                "This has yet to be implemented, falling back to uniform probabilistic grammar."
            )
            pcfg = ProbDetGrammar.uniform(cfg)
            pass  # TODO
        pcfg.init_sampling(seed)
        rng = np.random.default_rng(seed)
        tasks = []
        pbar = tqdm.tqdm(total=gen_dataset_size, desc="tasks generated")
        generated = set()
        for __ in range(gen_dataset_size):
            worlds = [random_world(width, height, rng) for _ in range(grids)]
            program = pcfg.sample_program()
            i = 0
            while program in generated:
                program = pcfg.sample_program()
                i += 1
                assert (
                    i < 10000
                ), f"Grammar is likely too shallow to only generate unique programs"

            task = Task(
                tr,
                PBE(
                    [
                        Example([worlds[i]], evaluator.eval(program, [worlds[i]]))
                        for i in range(grids)
                    ]
                ),
            )
            pbar.update(1)
            tasks.append(task)
            if len(tasks) == gen_dataset_size:
                break
        pbar.close()
        gen_dataset = Dataset(
            tasks,
            {
                "seed": seed,
                "max_depth": max_depth,
                "dsl": "karel",
            },
        )
        print("done in", c.elapsed_time(), "s")
    print("Saving dataset...", end="", flush=True)
    with chrono.clock("dataset.save") as c:
        gen_dataset.save(output_file)
        print("done in", c.elapsed_time(), "s")

    # ================================
    # Print some stats
    # ================================
    # Generate the CFG dictionnary
    all_type_requests = gen_dataset.type_requests()
    print(f"{len(all_type_requests)} type requests supported.")
