import json
from json.decoder import JSONDecodeError

import regex as re
from langchain.pydantic_v1 import ValidationError
from langchain.schema import OutputParserException


def json_parser(llm_output) -> dict:
    try:
        # Greedy search for 1st json candidate.
        match = re.search(
            r"\{.*\}", llm_output.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        json_str = ""
        if match:
            json_str = match.group()
        llm_output = json.loads(json_str, strict=False)
        return llm_output
    except (JSONDecodeError, ValidationError) as e:
        msg = f"Failed to parse response from completion {llm_output}. Got: {e}"
        raise OutputParserException(msg, llm_output=llm_output)
