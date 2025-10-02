import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pprint import pprint
from typing import Literal, List, Dict, Tuple, Any
import random
import ast
import openai
import pandas as pd
from tqdm import tqdm

from webscale_rl.behavior_template.checker import CHECKER_TEMPLATE, CHECKER_FORMAT
from webscale_rl.behavior_template.generator import GENERATOR_TEMPLATE, GENERATOR_FORMAT
from webscale_rl.behavior_template.identifier import IDENTIFIER_TEMPLATE, IDENTIFIER_FORMAT, ALL_DOMAINS
from webscale_rl.behavior_template.filter import FILTER_TEMPLATE, FILTER_FORMAT
from webscale_rl.utils.config import ModelConfig
from webscale_rl.utils.misc import clean_return_message

class BatchDataPipelineAgent:
    def __init__(
        self, 
        filter_cfg: ModelConfig,
        identifier_cfg: ModelConfig,
        generator_cfg: ModelConfig, 
        checker_cfg: ModelConfig,
        every_n_save_to_parquet: int = 500,
        batch_temp_dir: str = "batch_temp",
    ) -> None:
        self.filter_cfg = filter_cfg
        self.identifier_cfg = identifier_cfg
        self.generator_cfg = generator_cfg
        self.checker_cfg = checker_cfg

        # Single client is sufficient for batch processing
        self.client = self._build_client(filter_cfg.model_name, filter_cfg.port)
        self.model_name = generator_cfg.model_name

        self.logger = self.setup_logger()
        self.json_write_count = 0
        self.every_n_save_to_parquet = every_n_save_to_parquet
        
        # Create temp directory for batch files
        self.batch_temp_dir = batch_temp_dir
        if not os.path.exists(batch_temp_dir):
            os.makedirs(batch_temp_dir)

    def setup_logger(self, log_base_dir="logs"):
        """Sets up a logger for the agent with date and time in the file name."""
        if not os.path.exists(log_base_dir):
            os.mkdir(log_base_dir)
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_name = f'Batch_pretrain2rl_log_{current_time}.log'
        log_file_path = os.path.join(log_base_dir, log_file_name)
        logger = logging.getLogger(f"{self.model_name}_batch")
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def _build_client(self, model_name, port):
        from openai import OpenAI
        if "gpt" in model_name:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif model_name == "deepseek-chat":
            client = OpenAI(api_key=os.getenv("DS_API_KEY"), base_url="https://api.deepseek.com")
        else:
            base_url = f"http://localhost:{port}/v1"
            client = OpenAI(api_key="Empty", base_url=base_url)
        return client

    def create_batch_requests(self, requests: List[Dict], stage: str) -> str:
        """Create JSONL file for batch processing"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{stage}_batch_{timestamp}.jsonl"
        filepath = os.path.join(self.batch_temp_dir, filename)
        
        with open(filepath, 'w') as f:
            for i, request in enumerate(requests):
                batch_request = {
                    "custom_id": f"{stage}-{i}",
                    "method": "POST", 
                    "url": "/v1/chat/completions",
                    "body": request
                }
                f.write(json.dumps(batch_request) + '\n')
        
        return filepath

    def submit_batch(self, jsonl_file: str, stage: str) -> str:
        """Submit batch job and return batch ID"""
        # Upload file
        with open(jsonl_file, "rb") as f:
            batch_file = self.client.files.create(file=f, purpose="batch")
        
        # Create batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{stage} batch processing"}
        )
        
        self.logger.info(f"Submitted {stage} batch: {batch_job.id}")
        return batch_job.id

    def wait_for_batch(self, batch_id: str, stage: str) -> Dict:
        """Wait for batch completion and return results"""
        self.logger.info(f"Waiting for {stage} batch {batch_id}...")
        
        while True:
            batch = self.client.batches.retrieve(batch_id)
            
            if batch.status == "completed":
                self.logger.info(f"{stage} batch completed successfully")
                return batch
            elif batch.status == "failed":
                self.logger.error(f"{stage} batch failed: {batch}")
                raise Exception(f"Batch {batch_id} failed")
            elif batch.status in ["expired", "cancelled"]:
                self.logger.error(f"{stage} batch {batch.status}")
                raise Exception(f"Batch {batch_id} {batch.status}")
            
            # Wait before checking again
            time.sleep(10)

    def parse_batch_results(self, batch: Dict) -> List[Dict]:
        """Parse batch results from completed batch"""
        if not batch.output_file_id:
            return []
            
        # Download results
        result_content = self.client.files.content(batch.output_file_id)
        
        # Parse JSONL results
        results = []
        for line in result_content.content.decode('utf-8').strip().split('\n'):
            if line:
                result = json.loads(line)
                results.append(result)
        
        return results

    def run_batch(self, materials: List[str]) -> Tuple[List, List, List]:
        """Process materials using batch API"""
        self.logger.info(f"Starting batch processing for {len(materials)} materials")
        
        # Stage 1: Filter
        filter_results, filter_mapping = self._batch_filter(materials)
        
        # Stage 2: Identifier  
        identifier_results, identifier_mapping = self._batch_identifier(filter_results)
        
        # Stage 3: Generator
        generator_results, generator_mapping = self._batch_generator(identifier_results)
        
        # Stage 4: Checker
        final_results = self._batch_checker(generator_results, generator_mapping)
        
        # Organize results by original material
        augmented_data = []
        pass_list = []
        failure_list = []
        
        # Create results for each original material
        # for i, material in enumerate(materials):
        #     material_results = [r for r in final_results if r.get('original_material_idx') == i]
            
        #     if material_results:
        #         material_augmented = []
        #         material_pass = []
        #         material_failures = []
                
        #         for result in material_results:
        #             if result.get('success'):
        #                 material_augmented.append(result['data'])
        #                 material_pass.append(True)
        #                 material_failures.append(None)
        #             else:
        #                 material_augmented.append(None)
        #                 material_pass.append(False)
        #                 material_failures.append(result.get('error'))
                
        #         augmented_data.append(material_augmented)
        #         pass_list.append(material_pass)
        #         failure_list.append(material_failures)
        #     else:
        #         # Material failed at filter stage
        #         augmented_data.append([None])
        #         pass_list.append([False])
        #         failure_list.append([{"error_type": "filter_fail", "original_material": material}])
        for i, result in enumerate(final_results):
            if result.get('success'):
                augmented_data.append(result['data'])
                pass_list.append(True)
                failure_list.append(None)
            else:
                augmented_data.append(None)
                pass_list.append(False)
                failure_list.append(result.get('error', {}))
        
        self.logger.info(f"Batch processing completed")
        return augmented_data, pass_list, failure_list

    def _batch_filter(self, materials: List[str]) -> Tuple[List[Dict], Dict]:
        """Batch process filter stage"""
        requests = []
        for i, material in enumerate(materials):
            prompt = FILTER_TEMPLATE.format(
                material=material,
                format_inst=FILTER_FORMAT,
            )
            request = {
                "model": self.filter_cfg.model_name,
                "temperature": self.filter_cfg.temperature,
                "max_tokens": self.filter_cfg.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            requests.append(request)
        
        # Submit batch
        jsonl_file = self.create_batch_requests(requests, "filter")
        batch_id = self.submit_batch(jsonl_file, "filter")
        batch = self.wait_for_batch(batch_id, "filter")
        results = self.parse_batch_results(batch)
        
        # Process results
        passed_materials = []
        mapping = {}
        
        for result in results:
            idx = int(result['custom_id'].split('-')[1])
            material = materials[idx]
            
            try:
                content = result['response']['body']['choices'][0]['message']['content']
                clean_content = clean_return_message(content)
                parsed_result = ast.literal_eval(clean_content)
                
                if parsed_result.get("qualified") == "Y":
                    passed_materials.append({
                        'material': material,
                        'original_idx': idx
                    })
                    mapping[len(passed_materials) - 1] = idx
            except Exception as e:
                self.logger.error(f"Filter parsing error for material {idx}: {e}")
        
        return passed_materials, mapping

    def _batch_identifier(self, materials: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Batch process identifier stage"""
        requests = []
        all_domains = ', '.join(ALL_DOMAINS)
        
        for i, mat_data in enumerate(materials):
            prompt = IDENTIFIER_TEMPLATE.format(
                material=mat_data['material'],
                all_domains=all_domains,
                format_inst=IDENTIFIER_FORMAT,
            )
            request = {
                "model": self.identifier_cfg.model_name,
                "temperature": self.identifier_cfg.temperature,
                "max_tokens": self.identifier_cfg.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            requests.append(request)
        
        # Submit batch
        jsonl_file = self.create_batch_requests(requests, "identifier")
        batch_id = self.submit_batch(jsonl_file, "identifier")
        batch = self.wait_for_batch(batch_id, "identifier")
        results = self.parse_batch_results(batch)
        
        # Process results
        passed_materials = []
        mapping = {}
        
        for result in results:
            idx = int(result['custom_id'].split('-')[1])
            mat_data = materials[idx]
            
            try:
                content = result['response']['body']['choices'][0]['message']['content']
                clean_content = clean_return_message(content)
                parsed_result = ast.literal_eval(clean_content)
                
                if (parsed_result.get("domain") in ALL_DOMAINS and 
                    "persona" in parsed_result):
                    
                    personas = parsed_result["persona"].split(",")
                    for persona in personas:
                        passed_materials.append({
                            'material': mat_data['material'],
                            'domain': parsed_result["domain"],
                            'persona': persona.strip(),
                            'original_idx': mat_data['original_idx']
                        })
                        mapping[len(passed_materials) - 1] = mat_data['original_idx']
                        
            except Exception as e:
                self.logger.error(f"Identifier parsing error for material {idx}: {e}")
        
        return passed_materials, mapping

    def _batch_generator(self, materials: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Batch process generator stage"""
        requests = []
        
        for i, mat_data in enumerate(materials):
            prompt = GENERATOR_TEMPLATE.format(
                material=mat_data['material'],
                persona=mat_data['persona'],
                format_inst=GENERATOR_FORMAT,
            )
            request = {
                "model": self.generator_cfg.model_name,
                "temperature": self.generator_cfg.temperature,
                "max_tokens": self.generator_cfg.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            requests.append(request)
        
        # Submit batch
        jsonl_file = self.create_batch_requests(requests, "generator")
        batch_id = self.submit_batch(jsonl_file, "generator")
        batch = self.wait_for_batch(batch_id, "generator")
        results = self.parse_batch_results(batch)
        
        # Process results
        passed_materials = []
        mapping = {}
        
        for result in results:
            idx = int(result['custom_id'].split('-')[1])
            mat_data = materials[idx]
            
            try:
                content = result['response']['body']['choices'][0]['message']['content']
                clean_content = clean_return_message(content)
                parsed_result = ast.literal_eval(clean_content)
                
                if ("question" in parsed_result and "answer" in parsed_result):
                    passed_materials.append({
                        'material': mat_data['material'],
                        'domain': mat_data['domain'],
                        'persona': mat_data['persona'],
                        'question': parsed_result['question'],
                        'answer': parsed_result['answer'],
                        'original_idx': mat_data['original_idx']
                    })
                    mapping[len(passed_materials) - 1] = mat_data['original_idx']
                        
            except Exception as e:
                self.logger.error(f"Generator parsing error for material {idx}: {e}")
        
        return passed_materials, mapping

    def _batch_checker(self, materials: List[Dict], mapping: Dict) -> List[Dict]:
        """Batch process checker stage"""
        requests = []
        
        for i, mat_data in enumerate(materials):
            prompt = CHECKER_TEMPLATE.format(
                material=mat_data['material'],
                question=mat_data['question'],
                answer=mat_data['answer'],
                format_inst=CHECKER_FORMAT,
            )
            request = {
                "model": self.checker_cfg.model_name,
                "temperature": self.checker_cfg.temperature,
                "max_tokens": self.checker_cfg.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            requests.append(request)
        
        # Submit batch
        jsonl_file = self.create_batch_requests(requests, "checker")
        batch_id = self.submit_batch(jsonl_file, "checker")
        batch = self.wait_for_batch(batch_id, "checker")
        results = self.parse_batch_results(batch)
        
        # Process results
        final_results = []
        
        for result in results:
            idx = int(result['custom_id'].split('-')[1])
            mat_data = materials[idx]
            
            try:
                content = result['response']['body']['choices'][0]['message']['content']
                clean_content = clean_return_message(content)
                parsed_result = ast.literal_eval(clean_content)
                
                if (parsed_result.get("answer_correctness") == "Y" and 
                    parsed_result.get("info_leakage") == "N"):
                    
                    final_results.append({
                        'success': True,
                        'original_material_idx': mat_data['original_idx'],
                        'data': {
                            "original_material": mat_data['material'],
                            "question": mat_data['question'],
                            "answer": mat_data['answer'],
                            "domain": mat_data['domain'],
                            "persona": mat_data['persona'],
                        }
                    })
                else:
                    final_results.append({
                        'success': False,
                        'original_material_idx': mat_data['original_idx'],
                        'error': parsed_result
                    })
                        
            except Exception as e:
                self.logger.error(f"Checker parsing error for material {idx}: {e}")
                final_results.append({
                    'success': False,
                    'original_material_idx': mat_data['original_idx'],
                    'error': {
                        "error_type": "checker_parsing_error",
                        "error_message": str(e)
                    }
                })
        
        return final_results

    def write(
        self, 
        result, 
        file_to_open, 
        base_dir="./datasets/parallel", 
        prefix="",
        save_to_parquet: bool = False,
    ):
        """Same write method as original agent for compatibility"""
        if not save_to_parquet:
            if not os.path.exists(os.path.join(base_dir, self.model_name)):
                os.makedirs(os.path.join(base_dir, self.model_name))
            if not isinstance(result, list):
                result = [result]
            
            file_path = os.path.join(base_dir, self.model_name, prefix + file_to_open)
            with open(file_path, "a+") as f:
                for res in result:
                    try:
                        f.write(json.dumps(res) + "\n")
                    except Exception as e:
                        print(e)
                        print(f"Object {res} is not serializable.")
        
        else:
            parquet_name = os.path.splitext(file_to_open)[0] + ".parquet"
            parquet_path = os.path.join(base_dir, self.model_name, prefix + parquet_name)
            try:
                df_to_write = pd.DataFrame(result)
                if os.path.exists(parquet_path):
                    df_existing = pd.read_parquet(parquet_path)
                    df_to_write = pd.concat([df_existing, df_to_write], ignore_index=True)
                df_to_write.to_parquet(parquet_path, index=False)
            except Exception as e:
                self.logger.error(f"Error writing to parquet: {e}", exc_info=True)

    def cleanup_temp_files(self):
        """Clean up temporary batch files"""
        try:
            import shutil
            if os.path.exists(self.batch_temp_dir):
                shutil.rmtree(self.batch_temp_dir)
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")