def process_query(query_text: str) -> dict:
    tokens = process_raw(query_text)
    processed = ' '.join(tokens)
    return {
        'raw': query_text,
        'tokenized': tokens,
        'processed': processed,
    }