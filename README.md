Este laboratório foi desenvolvido por Andreas com auxílio de IA em pontos de lógica, organização do código e dúvidas específicas sobre uso de VRAM, KV Cache, FlashAttention-2 e fallback para SDPA. A implementação foi revisada e validada localmente, mas o benchmark completo com GPU depende de outro ambiente porque minha máquina atual não possui CUDA disponível para executar esses modelos e medir VRAM real.

# Laboratório 10 - RAG, QLoRA e otimização de inferência

Este repositório contém minha implementação do Lab 10. A ideia é reproduzir o problema do enunciado de forma controlada: carregar um modelo causal quantizado em 4 bits, montar um contexto médico longo como se viesse de um RAG, gerar 100 tokens sem KV Cache e depois repetir com KV Cache.

Um cuidado importante ficou explícito no código: `flash_attention_2` só é usado quando o ambiente realmente suporta FlashAttention-2. Se a GPU for anterior a Ampere, como uma T4 do Colab free, se não houver CUDA, ou se o pacote `flash-attn` não estiver instalado, o backend usado passa a ser `sdpa`, o Scaled Dot Product Attention do PyTorch. Assim o laboratório continua executável sem esconder a limitação do hardware.

## Arquivos

- `lab10_attention.py`: decide se o backend será `flash_attention_2` ou `sdpa`.
- `lab10_pipeline.py`: executa o benchmark com modelo quantizado, contexto longo e comparação sem/com KV Cache.
- `tests/test_attention_backend.py`: valida o fallback em cenários como CPU, T4 e GPU Ampere.
- `requirements.txt`: dependências principais do experimento.

## Como testar

```bash
py -m unittest discover
```

Os testes não baixam modelo e não exigem GPU. Eles validam:

- fallback de FlashAttention-2 para SDPA em CPU, T4 e ambientes sem `flash-attn`;
- uso de GPU Ampere+ quando o pacote `flash-attn` está disponível;
- geração de exatamente 100 tokens no benchmark;
- alternância entre `use_cache=False` e `use_cache=True`;
- formato do JSON de métricas salvo pelo pipeline.

## Como executar o benchmark

Instale as dependências em um ambiente com GPU:

```bash
py -m pip install -r requirements.txt
```

Depois execute:

```bash
py lab10_pipeline.py --target-context-tokens 12000 --max-new-tokens 100
```

Para forçar o fallback SDPA, útil em Colab free/T4:

```bash
py lab10_pipeline.py --disable-flash --target-context-tokens 12000 --max-new-tokens 100
```

O resultado é salvo em `outputs/metrics.json`, contendo:

- VRAM usada após carregar o modelo quantizado.
- Tempo e pico de VRAM com `use_cache=False`.
- Tempo e pico de VRAM com `use_cache=True`.
- Backend de atenção escolhido e justificativa.

## Métricas do benchmark

Preencher após a execução no ambiente de GPU usado na entrega final:

| Configuração | Tempo de geração | Pico de VRAM | Observação |
| --- | ---: | ---: | --- |
| Modelo 4-bit carregado | - | - | VRAM inicial após quantização |
| Sem KV Cache | - | - | Recalcula Q, K e V a cada token |
| Com KV Cache + backend escolhido | - | - | Usa cache e `flash_attention_2` ou `sdpa` |

## Validação local

Validação feita neste computador em 28/05/2026:

```text
PyTorch: 2.11.0+cpu
CUDA disponível: False
Backend detectado: sdpa
Motivo: CUDA indisponível; usando PyTorch SDPA como fallback.
```

Testes executados:

```text
py -m py_compile lab10_attention.py lab10_pipeline.py tests/test_attention_backend.py tests/test_pipeline_core.py
OK

unittest discover
Ran 7 tests
OK
```

O benchmark real de VRAM ainda precisa ser executado em um ambiente com GPU CUDA e `bitsandbytes`, porque esta máquina local está com PyTorch CPU e não consegue medir `torch.cuda.max_memory_allocated()` de forma válida.

## Checklist do enunciado

| Requisito | Onde está |
| --- | --- |
| Modelo causal autoregressivo | `AutoModelForCausalLM.from_pretrained` em `lab10_pipeline.py` |
| QLoRA / carga 4-bit | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` |
| Contexto médico fictício longo | `build_fake_medical_context(12000)` |
| Tokenização com `AutoTokenizer` | `AutoTokenizer.from_pretrained` e `tokenizer(prompt, return_tensors="pt")` |
| Geração de 100 tokens | argumento padrão `--max-new-tokens 100` |
| Teste sem cache | chamada com `use_cache=False` |
| Teste com KV Cache | chamada com `use_cache=True` |
| FlashAttention-2 | backend `flash_attention_2` quando GPU/pacote suportam |
| Fallback seguro | backend `sdpa` quando FlashAttention-2 não é viável |
| Métricas | `outputs/metrics.json` |

## Parecer técnico

A combinação de QLoRA, KV Cache e FlashAttention atua em gargalos diferentes do Transformer. A quantização em 4 bits reduz o custo fixo de carregar os pesos do modelo na VRAM, permitindo iniciar a inferência com um modelo que seria bem mais caro em FP16. O KV Cache evita recalcular chaves e valores para todos os tokens anteriores a cada novo token gerado, o que reduz muito o custo incremental do decoder. Já o FlashAttention-2 reorganiza o cálculo da atenção para usar melhor a hierarquia de memória da GPU, diminuindo leituras e escritas intermediárias na VRAM. Quando o hardware não atende aos requisitos do FlashAttention-2, o fallback para SDPA mantém a execução correta e ainda usa a implementação otimizada do PyTorch.

Mesmo assim, um contexto de 2 milhões de tokens ultrapassaria o limite prático desse desenho. FlashAttention reduz o desperdício de memória no cálculo da atenção, mas não muda o fato de que o Transformer ainda precisa lidar com dependências globais em uma sequência gigantesca e manter estruturas proporcionais ao tamanho do contexto, especialmente durante prompting e cache. Nesse cenário, a indústria passa a considerar arquiteturas como State Space Models, por exemplo Mamba, porque elas modelam sequências longas com estado recorrente compacto e complexidade de memória muito menor para inferência contínua. A troca não é apenas uma otimização de kernel; é uma mudança arquitetural para evitar que o comprimento da sequência domine a memória.
