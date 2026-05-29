from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class AttentionChoice:
    implementation: str
    reason: str
    flash_available: bool


def can_use_flash_attention_2(
    cuda_available: bool,
    capability: tuple[int, int] | None,
    flash_attn_installed: bool,
) -> bool:
    """Return True only for environments where FlashAttention-2 is realistic."""
    if not cuda_available or capability is None:
        return False

    major, _minor = capability
    return major >= 8 and flash_attn_installed


def choose_attention_implementation(
    prefer_flash: bool = True,
    cuda_available: bool = False,
    capability: tuple[int, int] | None = None,
    flash_attn_installed: bool = False,
) -> AttentionChoice:
    if not prefer_flash:
        return AttentionChoice(
            implementation="sdpa",
            reason="FlashAttention-2 desativado por configuracao; usando PyTorch SDPA.",
            flash_available=False,
        )

    flash_available = can_use_flash_attention_2(
        cuda_available=cuda_available,
        capability=capability,
        flash_attn_installed=flash_attn_installed,
    )
    if flash_available:
        return AttentionChoice(
            implementation="flash_attention_2",
            reason="GPU Ampere+ e pacote flash-attn detectados.",
            flash_available=True,
        )

    if not cuda_available:
        reason = "CUDA indisponivel; usando PyTorch SDPA como fallback."
    elif capability is None:
        reason = "Nao foi possivel identificar a capacidade da GPU; usando PyTorch SDPA."
    elif capability[0] < 8:
        reason = (
            f"GPU compute capability {capability[0]}.{capability[1]} nao atende "
            "ao requisito Ampere+ do FlashAttention-2; usando PyTorch SDPA."
        )
    elif not flash_attn_installed:
        reason = "Pacote flash-attn nao instalado; usando PyTorch SDPA."
    else:
        reason = "Ambiente nao compativel com FlashAttention-2; usando PyTorch SDPA."

    return AttentionChoice(
        implementation="sdpa",
        reason=reason,
        flash_available=False,
    )


def detect_attention_implementation(prefer_flash: bool = True) -> AttentionChoice:
    try:
        import torch
    except ImportError:
        return choose_attention_implementation(prefer_flash=prefer_flash)

    cuda_available = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability() if cuda_available else None
    flash_attn_installed = find_spec("flash_attn") is not None

    return choose_attention_implementation(
        prefer_flash=prefer_flash,
        cuda_available=cuda_available,
        capability=capability,
        flash_attn_installed=flash_attn_installed,
    )
