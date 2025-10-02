"""
This script processes the data using openai batch API.
However, according to our testing, the batch API is not stable and it's significantly slower than the individual requests.
So we recommend using the `main.py` script instead.
"""

import argparse
import json
import os
import random
from pprint import pprint
import datasets
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from webscale_rl.agent.batch_agent import BatchDataPipelineAgent
from webscale_rl.utils.config import ModelConfig

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--port", type=int, default="8011")
    parser.add_argument("--seed_dataset_dir", type=str, default="../data/pretrain/Wikipedia_en")
    parser.add_argument("--augmented_dataset_dir", type=str, default="data/RL_datasets")
    parser.add_argument("--augmented_dataset_filename", type=str, default="wikipedia.jsonl")
    parser.add_argument("--augmented_failure_filename", type=str, default="failure_log.jsonl")
    parser.add_argument("--categories", type=str, default="math", help="Comma-separated list of categories or 'all' for all categories")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    
    # Batch processing parameters
    parser.add_argument("--batch_size", type=int, default=500, help="Number of materials to process in each batch")
    parser.add_argument("--use_batch", action="store_true", help="Use batch processing instead of individual requests")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Model configurations (same as original)
    filter_cfg = ModelConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port,
    )
    identifier_cfg = ModelConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port + 1
    )
    generator_cfg = ModelConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port + 2
    )
    checker_cfg = ModelConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port + 3
    )

    # Initialize batch agent
    agent = BatchDataPipelineAgent(filter_cfg, identifier_cfg, generator_cfg, checker_cfg, every_n_save_to_parquet=500)
    
    statistics = {
        "success_count": 0,
        "failure_count": 0,
        "total_count": 0,
    }

    # Load the pretrain data (same as original)
    files = os.listdir(args.seed_dataset_dir)
    files = [f for f in files if f.endswith(".json")]
    text_key = "text"

    start_idx, end_idx = 190, 193
    args.augmented_dataset_filename = f"{args.augmented_dataset_filename.split('.')[0]}_{start_idx}_{end_idx}.jsonl"

    # Load data into a list
    all_pretrain_data = []
    for file in files[start_idx:end_idx]:
        with open(os.path.join(args.seed_dataset_dir, file), "r") as f:
            for line in f:
                all_pretrain_data.append(json.loads(line))
    
    # random.shuffle(all_pretrain_data)
    # all_pretrain_data = all_pretrain_data[:200]  # Limit for testing
    
    # Extract materials
    materials = [data_item[text_key] for data_item in all_pretrain_data]
    
    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(materials) + batch_size - 1) // batch_size
    
    print(f"Processing {len(materials)} materials in {total_batches} batches of size {batch_size}")
    
    try:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(materials))
            batch_materials = materials[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(batch_materials)} materials)...")
            
            # Process batch
            augmented_data_batch, pass_list_batch, failure_list_batch = agent.run_batch(batch_materials)
            
            passed_data = []
            for augmented_data, pass_ok, failure in zip(augmented_data_batch, pass_list_batch, failure_list_batch):
                if pass_ok:
                    passed_data.append(augmented_data)
                else:
                    passed_data.append(failure)
            
            agent.write(passed_data, args.augmented_dataset_filename, base_dir=args.augmented_dataset_dir, save_to_parquet=True)

            # # Write results (same format as original)
            # for i, (augmented_data_list, pass_list, failure_list) in enumerate(zip(augmented_data_batch, pass_list_batch, failure_list_batch)):
            #     for augmented_data, pass_ok, failure in zip(augmented_data_list, pass_list, failure_list):
            #         if pass_ok:
            #             agent.write(augmented_data, args.augmented_dataset_filename, base_dir=args.augmented_dataset_dir, save_to_parquet=False)
            #             statistics["success_count"] += 1
            #         else:
            #             agent.write(failure, args.augmented_failure_filename, base_dir=args.augmented_dataset_dir)
            #             statistics["failure_count"] += 1
            #         statistics["total_count"] += 1
            
            print(f"Batch {batch_idx + 1} completed. Total processed: {statistics['success_count']} success, {statistics['failure_count']} failures.")
            
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        agent.logger.error(f"Error during batch processing: {e}", exc_info=True)
    finally:
        # Cleanup temporary files
        agent.cleanup_temp_files()
        print(f"\nFinal statistics: {statistics['success_count']} success, {statistics['failure_count']} failures, {statistics['total_count']} total")

if __name__ == "__main__":
    main()