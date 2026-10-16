"""Frozen FLAN-T5-Large encoder; same conditioning as upstream MiniT2I."""
from pathlib import Path
from .assets import validate_t5_assets


def load_text_encoder(path):
    path = Path(path)
    # Verify origin and the entire pinned checkpoint before any deserialization.
    validate_t5_assets(path, verify_payloads=True)
    import jax.numpy as jnp
    from transformers import AutoTokenizer, FlaxT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = FlaxT5EncoderModel.from_pretrained(
        str(path), from_pt=False,
        dtype=jnp.bfloat16, local_files_only=True)
    if model.config.d_model != 1024 or model.config.num_layers != 24:
        raise ValueError('Expected released FLAN-T5-Large encoder')
    def encode(params, ids, mask):
        return model(input_ids=ids, attention_mask=mask, params=params,
                     train=False).last_hidden_state
    def tokenize(captions):
        return tokenizer(captions, return_tensors='np', padding='max_length',
                         truncation=True, max_length=256)
    return encode, model.params, tokenize
