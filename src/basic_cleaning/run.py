#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Artifact downloaded: "+args.input_artifact)
    df = pd.read_csv(artifact_local_path)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    df.to_csv("clean_sample.csv", index=False)

    output_artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
            )
    output_artifact.add_file("clean_sample.csv")
    run.log_artifact(output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="W&B artifact containing a .csv with the raw data",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact containing the processed data",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="The type of the output artifact, maybe processed_data?",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum per-night-price of the considered apartments",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum per-night-price of the considered apartments",
        required=True
    )


    args = parser.parse_args()

    go(args)
