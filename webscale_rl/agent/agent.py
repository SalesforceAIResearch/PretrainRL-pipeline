import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Literal
import random
import ast
import openai
import pandas as pd

from webscale_rl.sampler.fewshot_example_sampler import FewShotExampleSampler
from webscale_rl.behavior_template.fewshot import (
    GENERATOR_FEW_SHOT_TEMPLATE,
)
from webscale_rl.behavior_template.checker import CHECKER_TEMPLATE, CHECKER_FORMAT
from webscale_rl.behavior_template.generator import GENERATOR_TEMPLATE, GENERATOR_FORMAT
from webscale_rl.behavior_template.identifier import IDENTIFIER_TEMPLATE, IDENTIFIER_FORMAT, ALL_DOMAINS
from webscale_rl.behavior_template.filter import FILTER_TEMPLATE, FILTER_FORMAT
from webscale_rl.utils.config import ModelConfig
from webscale_rl.utils.misc import clean_return_message


FEW_SHOT_LIBRARY_PATH = {
    "Math": "domain_specific_library/math.jsonl",
    "Technology & Engineering": "domain_specific_library/tech.jsonl",
    "Coding": "domain_specific_library/coding.jsonl",
    "Social Science": "domain_specific_library/social_science.jsonl",
    "Natural Science": "domain_specific_library/natural_science.jsonl",
    "Travel & Lifestyle": "domain_specific_library/travel.jsonl",
    "Commerce & Economics": "domain_specific_library/commerce.jsonl",
    "Medicine & Health": "domain_specific_library/medicine.jsonl",
    "Education": "domain_specific_library/education.jsonl",
    "Other": "domain_specific_library/other.jsonl",
}

assert set(ALL_DOMAINS) == set(FEW_SHOT_LIBRARY_PATH.keys()), "All domains must be in the few shot library path"


class DataPipelineAgent:
    def __init__(
        self, 
        filter_cfg: ModelConfig,
        identifier_cfg: ModelConfig,
        generator_cfg: ModelConfig, 
        checker_cfg: ModelConfig,
        use_persona: bool = True,
        domain: str = "", # if use_persona is False, you need to specify the domain, and the persona will be the domain + " instructor"
        every_n_save_to_parquet: int = 2000,
    ) -> None:
        self.filter_cfg = filter_cfg
        self.identifier_cfg = identifier_cfg
        self.generator_cfg = generator_cfg
        self.checker_cfg = checker_cfg

        self.few_shot_exampler = FewShotExampleSampler(
            few_shot_example_template=GENERATOR_FEW_SHOT_TEMPLATE,
            example_data_path=FEW_SHOT_LIBRARY_PATH,
        )

        self.filter = self._build_client(filter_cfg.model_name, filter_cfg.port)
        self.identifier = self._build_client(identifier_cfg.model_name, identifier_cfg.port)
        self.generator = self._build_client(generator_cfg.model_name, generator_cfg.port)
        self.checker = self._build_client(checker_cfg.model_name, checker_cfg.port)
        
        self.model_name = generator_cfg.model_name

        self.logger = self.setup_logger()
        self.json_write_count = 0
        self.use_persona = use_persona
        self.predefined_domain = domain
        self.every_n_save_to_parquet = every_n_save_to_parquet # flush to parquet every N records

    def setup_logger(self, log_base_dir="logs"):
        """Sets up a logger for the agent with date and time in the file name."""
        if not os.path.exists(log_base_dir):
            os.mkdir(log_base_dir)
        # Get current date and time for the log file name
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_name = f'{self.model_name}_error_log_{current_time}.log'
        log_file_path = os.path.join(log_base_dir, log_file_name)
        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def _build_client(self, model_name, port):
        from openai import OpenAI
        if "gpt" in model_name:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # salesforce research api
            # client = openai.OpenAI(
            #     base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
            #     api_key="dummy",
            #     default_headers = {"X-Api-Key": os.getenv("X_API_KEY")}
            # )
        else:
            base_url = f"http://localhost:{port}/v1"
            client = OpenAI(api_key="Empty", base_url=base_url)
        return client
    

    def get_few_shot_example(
        self, 
        num_fewshot: int,
        behavior_type: str,
    ) -> str:
        examples = self.few_shot_exampler.sample(num_fewshot, behavior_type)
        formatted_examples = self.few_shot_exampler.format_examples(examples)
        
        return formatted_examples

    def inference(self, prompt: str, client_type: Literal["generator", "checker", "filter", "identifier"] = "generator"):
        if client_type == "generator":
            client = self.generator
            cfg = self.generator_cfg
        elif client_type == "checker":
            client = self.checker
            cfg = self.checker_cfg
        elif client_type == "filter":
            client = self.filter
            cfg = self.filter_cfg
        elif client_type == "identifier":
            client = self.identifier
            cfg = self.identifier_cfg
            
        messages = [{"role": "user", "content": prompt}]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = client.chat.completions.create(
                    messages=messages,
                    model=cfg.model_name,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                latency = time.time() - start_time
                result = response.choices[0].message.content
                metadata = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "latency": latency,
                }
                return result, metadata
            except openai.RateLimitError as e:
                wait = (2 ** attempt) + random.random()
                self.logger.warning(f"RateLimitError on attempt {attempt+1}/{max_retries}, retrying in {wait:.2f}s", exc_info=True)
                time.sleep(wait)
            except Exception as e:
                return None, {
                    "error_type": str(e),
                    "error_message": "An unknown error occurred during inference.",
                }

    def run(self, material: str) -> tuple[list, list, list]:
        # 1. filter the material
        filter_prompt = FILTER_TEMPLATE.format(
            material=material,
            format_inst=FILTER_FORMAT,
        )
        filter_results, metadata = self.inference(filter_prompt, client_type="filter")
        try:
            filter_results = clean_return_message(filter_results)
            filter_results = ast.literal_eval(filter_results)
            assert "qualified" in filter_results
        except Exception as e:
            if filter_results is None:
                error_info = f"Error parsing the filter_results: {metadata['error_type']}"
            else:
                error_info = f"Error parsing the filter_results: {filter_results}"
            self.logger.error(error_info, exc_info=True)
            return [None], [False], [{
                "input_prompt": filter_prompt,
                "error_type": "filter_parsing_error",
                "error_message": error_info,
            }]
        
        if filter_results["qualified"] == "N":
            return [None], [False], [{
                "error_type": "filter_fail",
                "input_prompt": filter_prompt,
                "original_material": material,
            }]

        all_domains = ', '.join(ALL_DOMAINS)

        # 2. identify the domain and persona of the material
        if self.use_persona:
            identifier_prompt = IDENTIFIER_TEMPLATE.format(
                material=material,
                all_domains=all_domains,
                format_inst=IDENTIFIER_FORMAT,
            )
            identifier_results, metadata = self.inference(identifier_prompt, client_type="identifier")
            try:
                identifier_results = clean_return_message(identifier_results)
                identifier_results = ast.literal_eval(identifier_results)
                assert "domain" in identifier_results and "persona" in identifier_results
            except Exception as e:
                error_info = f"Error parsing the identifier_results: {identifier_results}"
                self.logger.error(error_info, exc_info=True)
                return [None], [False], [{
                    "input_prompt": identifier_prompt,
                    "error_type": "identifier_parsing_error",
                    "error_message": error_info,
                }]
            if identifier_results["domain"] not in ALL_DOMAINS:
                return [None], [False], [{
                    "error_type": "identifier_domain_error",
                    "original_material": material,
                    "domain": identifier_results["domain"],
                }]

            domain = identifier_results["domain"]
            # there may be multiple personas, we need to choose one
            persona_list = identifier_results["persona"].split(",")
        else:
            domain = self.predefined_domain
            persona_list = [self.predefined_domain + " instructor"]

        # 3. convert material to QA pairs for each persona
        augmented_data = []
        pass_type_check = []
        semantic_failure_reason = []

        for persona in persona_list:
            few_shot_example = self.get_few_shot_example(self.generator_cfg.num_fewshot, domain)

            generator_prompt = GENERATOR_TEMPLATE.format(
                material=material,
                few_shot_example=few_shot_example,
                persona=persona.strip(),
                format_inst=GENERATOR_FORMAT,
            )
            
            generator_results, metadata = self.inference(generator_prompt, client_type="generator")
            try:
                generator_results = clean_return_message(generator_results)
                generator_results = ast.literal_eval(generator_results)
                assert "question" in generator_results and "answer" in generator_results
            except Exception as e:
                error_info = f"Error parsing the generator_results: {generator_results}"
                self.logger.error(error_info, exc_info=True)
                augmented_data.append(None)
                pass_type_check.append(False)
                semantic_failure_reason.append({
                    "input_prompt": generator_prompt,
                    "error_type": "generator_parsing_error",
                    "error_message": error_info,
                })
                continue

            # 4. check the quality of the QA pair
            checker_prompt = CHECKER_TEMPLATE.format(
                material=material,
                question=generator_results["question"],
                answer=generator_results["answer"],
                format_inst=CHECKER_FORMAT,
            )
            checker_results, metadata = self.inference(checker_prompt, client_type="checker")
            try:
                checker_results = clean_return_message(checker_results)
                checker_results = ast.literal_eval(checker_results)
                assert "answer_correctness" in checker_results and "info_leakage" in checker_results and "has_context" in checker_results
            except Exception as e:
                error_info = f"Error parsing the checker_results: {checker_results}"
                self.logger.error(error_info, exc_info=True)
                augmented_data.append(None)
                pass_type_check.append(False)
                semantic_failure_reason.append({
                    "input_prompt": checker_prompt,
                    "error_type": "checker_parsing_error",
                    "error_message": error_info,
                })
                continue

            if checker_results["answer_correctness"] == "Y" and checker_results["info_leakage"] == "N" and checker_results["has_context"] == "Y":
                augmented_data.append({
                    "original_material": material,
                    "question": generator_results["question"],
                    "answer": generator_results["answer"],
                    "domain": domain,
                    "persona": persona,
                })
                pass_type_check.append(True)
                semantic_failure_reason.append(None)
            else:
                augmented_data.append(None)
                pass_type_check.append(False)
                semantic_failure_reason.append(checker_results)

        return augmented_data, pass_type_check, semantic_failure_reason
        

    def write(
        self, 
        result, 
        file_to_open, 
        base_dir="", 
        prefix="",
        save_to_parquet: bool = False,
    ):
        # This method is used to write the result to the file.
        if not os.path.exists(os.path.join(base_dir, self.model_name)):
            os.makedirs(os.path.join(base_dir, self.model_name))
        if not isinstance(result, list):
            # make it to be a list
            result = [result]
        
        # write to JSON log
        file_path = os.path.join(base_dir, self.model_name, prefix + file_to_open)
        with open(file_path, "a+") as f:
            for res in result:
                try:
                    f.write(json.dumps(res) + "\n")
                except Exception as e:
                    print(e)
                    print(f"Object {res} is not serializable.")
        # update counter and optionally flush JSON to parquet
        self.json_write_count += len(result)
        if save_to_parquet and self.json_write_count >= self.every_n_save_to_parquet:
            parquet_name = os.path.splitext(file_to_open)[0] + ".parquet"
            parquet_path = os.path.join(base_dir, self.model_name, prefix + parquet_name)
            try:
                df_json = pd.read_json(file_path, lines=True)
                if os.path.exists(parquet_path):
                    df_existing = pd.read_parquet(parquet_path)
                    df_all = pd.concat([df_existing, df_json], ignore_index=True)
                else:
                    df_all = df_json
                df_all.to_parquet(parquet_path, index=False)
            except Exception as e:
                self.logger.error(f"Error writing to parquet: {e}", exc_info=True)
            # clear JSON log and reset counter
            open(file_path, 'w').close()
            self.json_write_count = 0

