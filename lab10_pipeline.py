import argparse
import json
import time
from pathlib import Path

from lab10_attention import detect_attention_implementation


def require_torch():
    import torch

    return torch


def build_fake_medical_context(target_tokens: int = 12000) -> str:
    paragraph = (
        "Capitulo de manual clinico: paciente adulto com diabetes mellitus tipo 2, "
        "hipertensao arterial sistemica e risco cardiovascular aumentado. "
        "A avaliacao recomenda anamnese dirigida, revisao medicamentosa, exame fisico, "
        "controle glicemico, funcao renal, adesao terapeutica e sinais de alerta. "
        "Condutas devem ser individualizadas conforme idade, comorbidades, exames "
        "laboratoriais, risco de hipoglicemia e disponibilidade de acompanhamento. "
    )
    words_per_paragraph = len(paragraph.split())
    repetitions = max(1, (target_tokens + words_per_paragraph - 1) // words_per_paragraph)
    return paragraph * repetitions


def cuda_memory_mb() -> float:
    torch = require_torch()
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**2)


def peak_cuda_memory_mb() -> float:
    torch = require_torch()
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def load_quantized_model(model_id: str, attn_implementation: str):
    torch = require_torch()
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
    )


def generate_once(model, tokenizer, input_ids, max_new_tokens: int, use_cache: bool) -> dict:
    torch = require_torch()
    model.config.use_cache = use_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    return {
        "use_cache": use_cache,
        "seconds": round(elapsed, 3),
        "peak_vram_mb": round(peak_cuda_memory_mb(), 2),
        "generated_tokens": int(output.shape[-1] - input_ids.shape[-1]),
    }


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    attention = detect_attention_implementation(prefer_flash=not args.disable_flash)
    print(f"Backend de atencao: {attention.implementation}")
    print(f"Motivo: {attention.reason}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = load_quantized_model(args.model_id, attention.implementation)
    model.eval()

    loaded_vram_mb = round(cuda_memory_mb(), 2)
    context = build_fake_medical_context(args.target_context_tokens)
    prompt = (
        "Use o contexto medico abaixo para gerar um resumo clinico objetivo.\n\n"
        f"{context}\n\nResumo clinico:"
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"].to(model.device)

    no_cache = generate_once(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=False,
    )
    with_cache = generate_once(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )

    metrics = {
        "model_id": args.model_id,
        "attention_implementation": attention.implementation,
        "attention_reason": attention.reason,
        "input_tokens": int(input_ids.shape[-1]),
        "loaded_vram_mb": loaded_vram_mb,
        "no_cache": no_cache,
        "with_cache": with_cache,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 10: RAG longo com QLoRA, KV cache e fallback SDPA.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-context-tokens", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--disable-flash", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
