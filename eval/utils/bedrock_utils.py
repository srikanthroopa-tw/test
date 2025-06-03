"""Helper utilities for working with Amazon Bedrock from Python notebooks"""

import os
import boto3


def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")
