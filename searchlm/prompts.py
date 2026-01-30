"""Shared prompts for LLM query generation"""

import re

SYSTEM_PROMPT = """
You are an expert at generating boolean search queries for a search engine.
You will be given a question in natural language and you need to generate a
boolean search query for it.
The boolean search query will be used in conjunction with a search engine to
retrieve relevant documents.
You should generate a query that is as specific as possible to the question,
and that will return the most relevant documents.
You should use the following operators: AND, OR, NOT.

Below a few basic query formats are shown:

AND and OR conjunctions.
query = '(Old AND Man) OR Stream'

+(includes) and -(excludes) operators.
query = '+Old +Man chef -fished'

phrase search.
query = '"eighty-four days"'

Think step by step and generate the query.

The output format should be as follows:
<think>
your reasoning here
</think>
<query>
generated query here
</query>
"""

USER_PROMPT_TEMPLATE = """
Translate the following question into a boolean search query:
{question}
"""


def create_chat_prompt(query_text: str) -> list[dict]:
    """
    Create formatted chat prompt for query generation.

    Args:
        query_text: The query text to format

    Returns:
        List of chat messages
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=query_text)},
    ]
    return messages


def extract_query_from_output(text: str) -> str:
    """
    Extract query from model output (between <query> tags).

    Args:
        text: Model output text

    Returns:
        Extracted query or original text if no tags found
    """
    match = re.search(r"<query>\s*(.*?)\s*</query>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
