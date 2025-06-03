import glob
import json
import logging
import os
import re
import time
from json import JSONEncoder
from pathlib import Path

import yaml

from eval.utils import (
    count_tokens,
    get_bedrock_client,
    get_cosine_similarity,
    get_rouge_l_score,
    is_amazon_model,
    parse_model_response,
)

# Set up logging
logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global constants
CONFIG_FILE_PATH = "config.yaml"


class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Exception):
            return {"error_type": o.__class__.__name__, "error_message": str(o)}
        return JSONEncoder.default(self, o)


def get_text_response_from_bedrock(
    prompt: str, inference_parameters: dict, model_id: str, bedrock_client
) -> dict:
    """Get text response from Bedrock model.

    Args:
        prompt: The input prompt for the model
        inference_parameters: Model-specific inference parameters
        model_id: The ID of the model to use
        bedrock_client: Initialized Bedrock client

    Returns:
        Dict containing the model response and metadata
    """
    logger.info(
        f"model_id={model_id}, prompt length is {len(prompt)} characters, {len(prompt.split())} words"
    )

    ret = {
        "exception": None,
        "prompt": prompt,
        "completion": None,
        "model_id": model_id,
        "time_taken_in_seconds": None,
        "completion_token_count": None,
        "prompt_token_count": None,
    }

    body = (
        {"inputText": prompt, "textGenerationConfig": inference_parameters}
        if is_amazon_model(model_id)
        else {"prompt": prompt} | inference_parameters
    )

    try:
        st = time.time()
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="*/*",
            body=json.dumps(body).encode("utf-8"),
        )
        ret["time_taken_in_seconds"] = time.time() - st

        resp_body = json.loads(response["body"].read().decode("utf-8"))
        ret.update(parse_model_response(model_id, resp_body))

    except Exception as e:
        logger.error(
            f"Exception when calling invoke_model, model_id={model_id}, exception={e}"
        )
        ret["exception"] = e

    return ret


def process_transcript(
    transcript_path: str, config: dict, bedrock_client, experiment: dict, rep: int
) -> None:
    """Process a single transcript file with the given experiment configuration.

    Args:
        transcript_path: Path to the transcript file
        config: Configuration dictionary
        bedrock_client: Initialized Bedrock client
        experiment: Experiment configuration
        rep: Current repetition number
    """
    transcript = Path(transcript_path).read_text()
    file_id = "_".join(os.path.basename(transcript_path).split("_")[:-1])

    # Derive golden summary path from transcript path
    golden_summary_fpath = str(transcript_path).replace(
        "_transcript.txt", "_golden_summary.txt"
    )

    logger.info(f"Processing transcript: {transcript_path}, file_id={file_id}")
    logger.info(f"Using golden summary: {golden_summary_fpath}")

    for model_info in experiment["model_list"]:
        model_name = model_info["model"]
        model = config["bedrock_models"].get(model_name)

        if not model:
            logger.error(f"Model {model_name} not found in bedrock_models")
            continue

        model["model_id"] = model_name
        prompt_template = Path(
            os.path.join(config["dir"]["prompts"], model_info["prompt_template"])
        ).read_text()
        prompt = prompt_template.format(transcript)
        prompt_token_count = count_tokens(prompt)

        if (prompt_token_count + 125) > model["context_length"]:
            logger.info(
                f"Cannot summarize with {model['model_id']} - prompt length {prompt_token_count} exceeds context length {model['context_length']}"
            )
            continue

        inference_parameters = config["inference_params"][model["inference_param_set"]]
        resp = get_text_response_from_bedrock(
            prompt, inference_parameters, model["model_id"], bedrock_client
        )

        if resp["exception"]:
            handle_error_response(resp, transcript_path, model["model_id"], config)
            continue

        process_successful_response(
            resp,
            prompt_token_count,
            experiment["name"],
            file_id,
            model,
            golden_summary_fpath,
            rep,
            config,
        )


def handle_error_response(
    resp: dict, transcript_path: str, model_id: str, config: dict
) -> None:
    """Handle error response from Bedrock API."""
    logger.error(f"Exception occurred for {transcript_path}, model_id={model_id}")
    if (
        hasattr(resp["exception"], "response")
        and resp["exception"].response["Error"]["Code"] == "ThrottlingException"
    ):
        handle_throttling_exception(resp, transcript_path, model_id, config)


def handle_throttling_exception(
    resp: dict, transcript_path: str, model_id: str, config: dict
) -> None:
    """Handle throttling exception with retries."""
    retries = 0
    while retries < config["max_retries"]:
        retries += 1
        sleep_time = (
            retries * config["prompt"]["very_large_prompt"]["sleep_time"]
            if resp.get("very_large_prompt")
            else retries * config["prompt"]["normal_prompt"]["sleep_time"]
        )
        time.sleep(sleep_time)

        logger.info(
            f"Retrying for {transcript_path}, model_id={model_id}, attempt {retries}"
        )
        resp = get_text_response_from_bedrock(
            resp["prompt"],
            resp["inference_parameters"],
            model_id,
            resp["bedrock_client"],
        )

        if not resp["exception"]:
            logger.info(f"Retry succeeded after {retries} attempts")
            return

    logger.error(f"Max retries ({config['max_retries']}) exceeded, giving up")


def process_successful_response(
    resp: dict,
    prompt_token_count: int,
    exp_name: str,
    file_id: str,
    model: dict,
    golden_summary_fpath: str,
    rep: int,
    config: dict,
) -> None:
    """Process successful response from Bedrock API."""
    if resp["prompt_token_count"] is None:
        resp["prompt_token_count"] = prompt_token_count

    resp["completion"] = re.sub(
        r"<reason>|</reason>|<reasons>|</reasons>", "", resp["completion"]
    ).strip()

    # Save completion
    save_completion(resp, exp_name, file_id, model["model_id"], rep, config)

    # Calculate metrics if golden summary exists
    if Path(golden_summary_fpath).is_file():
        calculate_metrics(resp, golden_summary_fpath)

    # Calculate cost and word count
    calculate_costs_and_metrics(resp, model, exp_name)

    # Save metrics
    save_metrics(resp, exp_name, file_id, model["model_id"], rep, config)


def save_completion(
    resp: dict, exp_name: str, file_id: str, model_id: str, rep: int, config: dict
) -> None:
    """Save completion to file."""
    dir_path = os.path.join(config["dir"]["completions"], exp_name, file_id)
    os.makedirs(dir_path, exist_ok=True)
    Path(os.path.join(dir_path, f"{file_id}_{model_id}_rep{rep + 1}.txt")).write_text(
        resp["completion"]
    )


def calculate_metrics(resp: dict, golden_summary_fpath: str) -> None:
    """Calculate ROUGE and cosine similarity metrics."""
    golden_summary = Path(golden_summary_fpath).read_text()
    resp["rouge_l_f1_score"] = get_rouge_l_score(golden_summary, resp["completion"])
    resp["cosine_similarity"] = get_cosine_similarity(
        golden_summary, resp["completion"]
    )


def calculate_costs_and_metrics(resp: dict, model: dict, exp_name: str) -> None:
    """Calculate costs and additional metrics."""
    resp["cost"] = model["prompt_token_pricing_per_million"] * (
        resp["prompt_token_count"] / 1000000
    ) + model["completion_token_pricing_per_million"] * (
        resp["completion_token_count"] / 1000000
    )
    resp["completion_word_count"] = len(resp["completion"].split())
    resp["experiment"] = exp_name


def save_metrics(
    resp: dict, exp_name: str, file_id: str, model_id: str, rep: int, config: dict
) -> None:
    """Save metrics to file."""
    dir_path = os.path.join(config["dir"]["metrics"], exp_name, file_id)
    os.makedirs(dir_path, exist_ok=True)
    logger.info(json.dumps(resp, indent=2))
    Path(os.path.join(dir_path, f"{file_id}_{model_id}_rep{rep + 1}.json")).write_text(
        json.dumps(resp, cls=CustomJSONEncoder, indent=2)
    )


def main():
    """Main execution function."""
    try:
        # Read configuration
        with open(CONFIG_FILE_PATH) as yaml_in:
            config = yaml.safe_load(yaml_in)
        logger.info(f"Configuration loaded from {CONFIG_FILE_PATH}")

        # Initialize Bedrock client
        bedrock_client = get_bedrock_client()

        # Get transcript files
        transcript_files = glob.glob(
            os.path.join(config["dir"]["raw"], "*", "*transcript.txt")
        )
        logger.info(f"Found {len(transcript_files)} transcript files")

        # Process each transcript
        for transcript_path in transcript_files:
            for experiment in config["experiments"]:
                for rep in range(experiment["reps"]):
                    process_transcript(
                        transcript_path, config, bedrock_client, experiment, rep
                    )

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
