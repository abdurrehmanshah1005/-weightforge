import argparse
import json

from .demo import run
from .providers import ManualLLMProvider


def main():
    parser = argparse.ArgumentParser(description="WeightForge CPU research demo")
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--random-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proposal", action="append", help="LLM JSON file, repeat once per step")
    args = parser.parse_args()
    provider = ManualLLMProvider(args.proposal) if args.proposal else None
    if args.proposal and len(args.proposal) != args.budget:
        parser.error("Supply one --proposal per budget step")
    result = run(args.out, args.budget, args.random_runs, args.seed, provider)
    print(
        json.dumps(
            {k: result[k] for k in ("before_validation", "after_validation", "final_holdout")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
