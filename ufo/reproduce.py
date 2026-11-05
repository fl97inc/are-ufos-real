""" CLI to reproduce UFO grokking modular division experiments """
import argparse
from pathlib import Path
from ufo.matrix import experiment_matrix
from ufo.results import write_summary
from ufo.training import experiment_name, train_experiment

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce UFO modular-division experiments")
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="auto")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List all 144 experiments")
    run = commands.add_parser("run", help="Run one or all experiments")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int, help="Run one experiment by its 1-based list index")
    selection.add_argument("--all", action="store_true", help="Run the complete 144-experiment matrix")
    run.add_argument("--confirm-all", action="store_true", help="Required with --all because matrix is expensive")
    run.add_argument("--overwrite", action="store_true", help="Rerun experiments whose CSV already exists")
    commands.add_parser("summarize", help="Write results to summary.csv")
    return parser


def selected_experiments(parser: argparse.ArgumentParser, args: argparse.Namespace):
    experiments = experiment_matrix()
    if args.all:
        if not args.confirm_all:
            parser.error("running all experiments requires --confirm-all")
        return experiments
    if args.index < 1 or args.index > len(experiments):
        parser.error(f"--index must be between 1 and {len(experiments)}")
    return [experiments[args.index -1]]

def main():
    parser = build_parser()
    args = parser.parse_args()
    experiments = experiment_matrix()
    output = Path(args.output)
    if args.command == "list":
        for index, config in enumerate(experiments, start=1):
            print(f"{index:3d} {experiment_name(config)}")
        return
    if args.command == "summarize":
        path = write_summary(output)
        print(f"Wrote {path}")
        return

    selected = selected_experiments(parser, args)
    for index, config in enumerate(selected, start=1):
        path = output / f"{experiment_name(config)}.csv"
        if path.exists() and not args.overwrite:
            print(f"[{index}/{len(selected)}] skip {path.name}")
            continue
        print(f"[{index}/{len(selected)}] run " f"{experiment_name(config)}")
        train_experiment(config, output, args.device)

if __name__ == "__main__":
    main()
