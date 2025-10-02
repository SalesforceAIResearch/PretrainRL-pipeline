from typing import List
import argparse
import json
import os
import random
import datasets
from datasets import load_dataset
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from webscale_rl.agent.agent import DataPipelineAgent
from webscale_rl.utils.config import ModelConfig

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--port", type=int, default="8011")
    parser.add_argument("--seed_dataset_dir", type=str, default="")
    parser.add_argument("--RL_dataset_save_dir", type=str, default="data/RL_datasets")
    parser.add_argument("--RL_dataset_filename", type=str, default="webscale_rl.jsonl")
    parser.add_argument("--failure_log_filename", type=str, default="failure_log.jsonl")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-fewshot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel worker threads")
    return parser.parse_args()

def process_results(success_results, info, agent: DataPipelineAgent, file_name, statistics, base_dir):
    if success_results:
        statistics["success_count"] += len(success_results)
        agent.write(success_results, file_name, base_dir)

    for failure_type, key in zip(["failure_format", "failure_execution", "failure_semantic"],
                                ["failure_format_count", "failure_execution_count", "failure_semantic_count"]):
        failures = info[failure_type]
        if failures:
            statistics[key] += len(failures)
            agent.write(failures, file_name, base_dir, suffix=failure_type)

def main():
    args = get_args()
    filter_cfg = ModelConfig(
        model_name=args.model, # use mini model if you want
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port,
    )
    identifier_cfg = ModelConfig(
        model_name=args.model, # use mini model if you want
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port + 1
    )
    generator_cfg = ModelConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port + 2,
        num_fewshot=args.num_fewshot,
    )
    checker_cfg = ModelConfig(
        model_name=args.model, # use mini model if you want
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        port=args.port + 3
    )

    # Load your own pretrain data here
    # here is an example
    all_pretrain_data: List[str] = [
        "Armond Duck Chief\nArmond Duck Chief is a Canadian singer and songwriter of country music, who was a Juno Award nominee for Aboriginal Album of the Year at the Juno Awards of 2016 for his album The One.\n\nA member of the Siksika Nation from Alberta, he released his debut album Country Groove in 2011. In addition to the Juno Award nomination, The One was a winner for Best Country Album, and Duck Chief for Best Songwriter, at the 2015 Indigenous Music Awards.\n\nReferences\n\nCanadian country singer-songwriters\nFirst Nations musicians\nSiksika Nation people\nMusicians from Alberta\nLiving people\nYear of birth missing (living people)",
        "Al Baqsh\nAl Baqsh () is a sub-district located in Radman Al Awad District, Al Bayda Governorate, Yemen.  Al Baqsh had a population of 2298  according to the 2004 census.\n\nReferences \n\nSub-districts in Radman Al Awad District",
        "Readymoney Cove\nReadymoney Cove (, meaning mineral house cove) is a sandy beach to the south of the harbour town of Fowey, Cornwall, England, United Kingdom. It is sheltered by cliffs close to the mouth of the River Fowey estuary and bounded, on one side, by the medieval part of the town of Fowey and, on the other, by St Catherine's Castle. The beach can be covered during spring tides. The beach is cleaned daily during high season, and a bathing platform is moored in the bay. There is a small shop with public toilet facilities both of which are open all year round. Dogs are banned between 10am and 6pm during July and August. Above the cove is the former coach house which was the home of author, Daphne du Maurier, for a few years during the Second World War.\nComedian Dawn French used to live in a house overlooking the cove.\n\nReferences\n\nExternal links \n\n\u2013 More information on Readymoney Cove & beach\n\nBeaches of Cornwall\nFowey\nCoves of Cornwall",
    ] * 50
    
    
    agent = DataPipelineAgent(
        filter_cfg, 
        identifier_cfg, 
        generator_cfg, 
        checker_cfg, 
        every_n_save_to_parquet=5000
    )
    statistics = {
        "success_count": 0,
        "failure_count": 0,
        "total_count": 0,
    }
    
    # Process items in parallel
    def process_item(data_item):
        return agent.run(data_item)


    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, d) for d in all_pretrain_data]
        for future in tqdm(as_completed(futures), total=len(futures)):
            augmented_data_list, pass_list, failure_list = future.result()
            for augmented_data, pass_ok, failure in zip(augmented_data_list, pass_list, failure_list):
                if pass_ok:
                    agent.write(augmented_data, args.RL_dataset_filename, base_dir=args.RL_dataset_save_dir, save_to_parquet=True)
                    statistics["success_count"] += 1
                else:
                    agent.write(failure, args.failure_log_filename, base_dir=args.RL_dataset_save_dir)
                    statistics["failure_count"] += 1
                statistics["total_count"] += 1
            if (statistics["total_count"] ) % 5000 == 0:
                agent.logger.info(f"Processed {statistics['success_count']} success, {statistics['failure_count']} failures.")

if __name__ == "__main__":
    main()