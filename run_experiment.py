#!/usr/bin/env python3
"""Experiment dispatcher.  The publication pipeline is intentionally explicit."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", nargs="?", choices=["sncs_mpu"])
    parser.add_argument("--pipeline", dest="pipeline_flag", choices=["sncs_mpu"])
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--input", default="datasets/averaged_health_fitness_dataset.csv")
    parser.add_argument("--output", default="artifacts/sncs_mpu")
    args = parser.parse_args()
    pipeline = args.pipeline_flag or args.pipeline
    if pipeline != "sncs_mpu":
        parser.error("select the sncs_mpu pipeline")
    if args.pipeline == "sncs_mpu" and args.run_label not in (None, "sncs_mpu"):
        parser.error("run label must be sncs_mpu")
    if pipeline == "sncs_mpu":
        from sncs_mpu import run
        run(args.input, args.output)


if __name__ == "__main__":
    main()
