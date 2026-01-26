"""Shared prompts for LLM query generation"""

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


def format_user_prompt(question: str) -> str:
    """
    Format user prompt with question.

    Args:
        question: The question to convert into a search query

    Returns:
        Formatted user prompt
    """
    return USER_PROMPT_TEMPLATE.format(question=question)
