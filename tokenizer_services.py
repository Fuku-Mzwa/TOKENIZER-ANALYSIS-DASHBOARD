# Hugging face tokenizer loading and tokenization utilities
# loads the tokenizer and produces token strings,
# which are then forwarded to the backend (efficiency_metrics.py,mbi.py)
# for metric computation
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Simple cache to avoid reloading the same tokenizer more than once
TOKENIZER_CACHE = {}


def load_tokenizer(model_id, use_fast=True):
    # Load a Hugging face tokenizer by its model name

    if model_id is None or model_id.strip() == "":
        raise ValueError("Tokenizer identifier cannot be empty.")

    model_id = model_id.strip()

    # If the tokenizer was already loaded, just reuse it
    if model_id in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[model_id]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=use_fast,
            trust_remote_code=True,
        )
        TOKENIZER_CACHE[model_id] = tokenizer
        logger.info("Loaded tokenizer: " + model_id)
        return tokenizer

    except Exception as e:
        logger.exception("Failed to load tokenizer " + model_id)
        error_message = (
            "Could not load tokenizer '" + model_id + "'."
            "Please check the Hugging Face model ID."
            "Error: " + str(e)
        )
        raise ValueError(error_message)


def clear_tokenizer_cache():
    # Wipe the cache, could be used when the user wants to force a fresh load
    TOKENIZER_CACHE.clear()


def tokenize_text(tokenizer, text):
    # Run the tokenizer on the piece of text and return results
    # Results will be used for metric calculations
    # The other parts of the application might need the token strings, their ids

    if text is None:
        text = ""
    text = str(text)

    try:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = encoded["input_ids"]

        # Some tokenizers return a nested list even for one input
        # If this happens, grab the first list inside it
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]

        tokens = tokenizer.convert_ids_to_tokens(ids)

    except Exception:
        # In case the tokenizer does not support the call above
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)

    byte_count = len(text.encode("utf-8"))

    result = {
        "tokens": tokens,
        "ids": ids,
        "token_count": len(tokens),
        "char_count": len(text),
        "byte_count": byte_count,
    }

    return result