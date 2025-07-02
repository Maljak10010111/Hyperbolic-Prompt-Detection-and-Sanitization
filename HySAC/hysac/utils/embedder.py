import torch
from tqdm import tqdm


def process_batch_embeddings(
    model_id,
    batch_idx,
    batch_size,
    all_prompts,
    all_categories,
    tokenizer,
    model,
    device,
    batch_cache_file,
    **kwargs,
):
    """
    Process a batch of prompts to generate and save embeddings.

    Args:
        batch_idx (int): Starting index of the batch
        batch_size (int): Size of the batch to process
        all_prompts (list): List of all text prompts
        all_categories (list): List of categories corresponding to prompts
        tokenizer: Tokenizer for text processing
        model: Model for generating embeddings
        device: Device to run computations on (cuda/cpu)
        batch_cache_file (str): Path to save the batch embeddings

    Returns:
        bool: True if batch processed successfully, False otherwise
    """
    print(f"Computing batch {batch_idx} to {batch_idx + batch_size}")

    batch_embeddings = []
    batch_prompts = all_prompts[batch_idx : batch_idx + batch_size]
    batch_categories = all_categories[batch_idx : batch_idx + batch_size]

    try:
        for prompt_idx, first_text_prompt in enumerate(tqdm(batch_prompts)):
            if "hysac" in model_id:
                embedding_result = _process_single_prompt_hysac(
                    first_text_prompt,
                    batch_categories[prompt_idx],
                    tokenizer,
                    model,
                    device,
                    batch_idx + prompt_idx,
                )
            else:
                embedding_result = _process_single_prompt_clip(
                    first_text_prompt,
                    batch_categories[prompt_idx],
                    tokenizer,
                    model,
                    device,
                    batch_idx + prompt_idx,
                    **kwargs,
                )

            if embedding_result is not None:
                batch_embeddings.append(embedding_result)
                

        # Save the batch embeddings
        torch.save(batch_embeddings, batch_cache_file)
        print(f"Saved batch {batch_idx} to {batch_cache_file}")
        return batch_embeddings

    except Exception as e:
        print(f"Error processing batch {batch_idx}:")
        print(f"Error details: {str(e)}")
        return None


def _process_single_prompt_hysac(
    text_prompt, category, tokenizer, model, device, global_idx
):
    """
    Process a single prompt to generate its embedding.

    Args:
        text_prompt (str): The text prompt to process
        category: The category associated with this prompt
        tokenizer: Tokenizer for text processing
        model: Model for generating embeddings
        device: Device to run computations on
        global_idx (int): Global index for error reporting

    Returns:
        tuple: (embedding, category) if successful, None if failed
    """
    try:
        # Tokenize the prompt
        text_prompt_tokens = tokenizer(
            text_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # Get input IDs and move to device
        text_prompt_tokens["input_ids"] = text_prompt_tokens["input_ids"].to(device)
        text_prompt_tokens["attention_mask"] = text_prompt_tokens["attention_mask"].to(device)
        # Generate embeddings with no gradient tracking
        with torch.no_grad():
            text_prompt_encoding = model.encode_text(text_prompt_tokens["input_ids"], project=True)
        # Process embeddings and move to CPU
        flattened_encoding = text_prompt_encoding.squeeze(0).to("cpu")
        # Clean up to free memory
        del text_prompt_tokens
        del text_prompt_encoding
        torch.cuda.empty_cache()
        if category is not None:
            return (flattened_encoding, category)
        else:
            return flattened_encoding

    except Exception as e:
        print(f"Error processing prompt {global_idx}: '{text_prompt}'")
        print(f"Error details: {str(e)}")
        return None


def _process_single_prompt_clip(
    text_prompt, category, tokenizer, model, device, global_idx, **kwargs
):
    """
    Process a single prompt to generate its embedding.

    Args:
        text_prompt (str): The text prompt to process
        category: The category associated with this prompt
        tokenizer: Tokenizer for text processing
        model: Model for generating embeddings
        device: Device to run computations on
        global_idx (int): Global index for error reporting

    Returns:
        tuple: (embedding, category) if successful, None if failed
    """
    try:
        # Tokenize the prompt
        text_prompt_tokens = tokenizer(
            text_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # Get input IDs and move to device
        text_input_ids = text_prompt_tokens["input_ids"].to(device)

        # Generate embeddings with no gradient tracking
        with torch.no_grad():
            text_prompt_encoding = model.encode_text(text_input_ids)

        # Process embeddings and move to CPU
        flattened_encoding = text_prompt_encoding.squeeze(0).to("cpu")

        # Clean up to free memory
        del text_input_ids
        del text_prompt_encoding
        torch.cuda.empty_cache()

        return (flattened_encoding, category)

    except Exception as e:
        print(f"Error processing prompt {global_idx}: '{text_prompt}'")
        print(f"Error details: {str(e)}")
        return None
