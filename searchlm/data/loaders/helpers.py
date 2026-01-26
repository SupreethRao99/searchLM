"""Shared helper functions for dataset loaders."""

from typing import Dict


def get_field_with_fallbacks(item: dict, *field_names: str, default: str = "") -> str:
    """
    Try multiple field names, return first non-empty value.
    
    Args:
        item: Dictionary to search
        *field_names: Field names to try in order
        default: Default value if none found
        
    Returns:
        First non-empty value found, or default
    """
    for field in field_names:
        value = item.get(field)
        if value:
            return str(value)
    return default


def filter_queries_by_qrels(all_queries: Dict[str, str], qrels: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Filter queries to only include those present in qrels.
    
    Args:
        all_queries: Dictionary mapping query_id -> query_text
        qrels: Dictionary of relevance judgments
        
    Returns:
        Filtered dictionary of queries
    """
    return {qid: text for qid, text in all_queries.items() if qid in qrels}
