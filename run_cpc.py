from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers.models.auto import configuration_auto as auto_config_module
from vllm import LLM
from vllm.sampling_params import SamplingParams

from prefix_constrained_logits_processor import PrefixConstrainedLogitsProcessor
from prefix_trie import CharacterPrefixTrie
from utils import _adapt_tokenizer, create_prefix_trie

auto_config_module.CONFIG_MAPPING.pop("aimv2", None)

class CPCGenerator:
    def __init__(self, model_name: str, default_max_new_tokens: int) -> None:
        self.model_name = model_name
        self.default_max_new_tokens = default_max_new_tokens
        self.tokenizer = _adapt_tokenizer(AutoTokenizer.from_pretrained(model_name))
        self.trie = create_prefix_trie(model_name)
        self.llm = LLM(
            model=model_name,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            max_model_len=4096,
            max_num_seqs=2,
            tensor_parallel_size=1,
        )

    def generate(self, prompt: str, character_prefix: str, max_new_tokens: Optional[int] = None):
        return generate_with_prefix_constraint(
            model_name=self.model_name,
            prompt=prompt,
            character_prefix=character_prefix,
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            tokenizer=self.tokenizer,
            trie=self.trie,
            llm=self.llm,
        )


def find_tokenization_breakpoint(tokenizer, trie: CharacterPrefixTrie, prefix: str):
    last_valid_position = 0
    for i in range(len(prefix), -1, -1):
        remaining = prefix[i:]
        if remaining and trie.find_all_tokens_starting_with(remaining):
            last_valid_position = i

    clean_prefix = prefix[:last_valid_position]
    remaining_prefix = prefix[last_valid_position:]
    clean_token_ids = tokenizer.encode(clean_prefix, add_special_tokens=False) if clean_prefix else []
    return clean_token_ids, remaining_prefix


def generate_with_prefix_constraint(
    model_name: str,
    prompt: str,
    character_prefix: str,
    max_new_tokens: int,
):
    tokenizer = _adapt_tokenizer(AutoTokenizer.from_pretrained(model_name))
    trie = create_prefix_trie(model_name)

    prefix_tokens, remaining_prefix = find_tokenization_breakpoint(tokenizer, trie, character_prefix)

    current_text = (
        prompt
        + tokenizer.decode(prefix_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if prefix_tokens
        else prompt
    )

    llm = LLM(
        model=model_name,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=1,
    )

    tokens_generated = 0

    while remaining_prefix and tokens_generated < max_new_tokens:
        processor = PrefixConstrainedLogitsProcessor(tokenizer, trie, remaining_prefix)
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=5,
            logits_processors=[processor],
        )

        outputs = llm.generate([current_text], sampling_params=sampling_params)
        new_text = outputs[0].outputs[0].text
        current_text += new_text
        tokens_generated += 1

        token_ids = tokenizer.encode(new_text, add_special_tokens=False)
        if not token_ids:
            break

        processor.update_remaining_prefix(token_ids[0])
        remaining_prefix = processor.remaining_prefix

    tokens_left = max_new_tokens - tokens_generated
    if tokens_left > 0 and not remaining_prefix:
        sampling_params = SamplingParams(max_tokens=tokens_left, temperature=0.0, top_p=1.0)
        outputs = llm.generate([current_text], sampling_params=sampling_params)
        current_text += outputs[0].outputs[0].text

    return current_text


class GenerateRequest(BaseModel):
    prompt: str
    character_prefix: Optional[str] = ""
    max_new_tokens: Optional[int] = None


class GenerateResponse(BaseModel):
    full_output: str
    generated_segment: str


def create_api_app() -> FastAPI:

    generator_holder = {"generator": None}

    def get_generator():
        if generator_holder["generator"] is None:
            generator_holder["generator"] = CPCGenerator("mistralai/Mistral-7B-Instruct-v0.3", 150)
        return generator_holder["generator"]

    app = FastAPI(title="Character Prefix Conditioning")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"ready": generator_holder["generator"] is not None}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
        generator = get_generator()
        limit = request.max_new_tokens
        full_output = generator.generate(
            prompt=request.prompt,
            character_prefix=request.character_prefix or "",
            max_new_tokens=limit,
        )
        generated_segment = full_output[len(request.prompt):]
        return GenerateResponse(full_output=full_output, generated_segment=generated_segment)

    return app
