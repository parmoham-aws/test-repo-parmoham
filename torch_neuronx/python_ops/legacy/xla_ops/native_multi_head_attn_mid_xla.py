"""XLA implementation of native_multi_head_attn_mid"""

import torch

from ...base import ExecutionResult
from .scaled_dot_product_attention_xla import ScaledDotProductAttnXLAImpl


class NativeMultiHeadAttnMidXLAImpl(ScaledDotProductAttnXLAImpl):
    """XLA implementation for Mid operation with transposed inputs"""

    def __init__(self, op_name: str):
        super().__init__(op_name)
        # Set flag to return single tensor instead of tuple
        self._returns_single_tensor = True

    def can_handle(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        use_causal_mask: bool = False,
        dropout_p: float = 0.0,
        training: bool = False,
        out: torch.Tensor | None = None,
    ) -> bool:
        """Check if XLA implementation can handle Mid operation parameters"""
        return super().can_handle(
            query,
            key,
            value,
            is_causal=use_causal_mask,
            dropout_p=dropout_p,
        )

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        use_causal_mask: bool = False,
        dropout_p: float = 0.0,
        training: bool = False,
        out: torch.Tensor | None = None,
    ) -> ExecutionResult:
        """Execute with input transposition for Mid format"""
        # Transpose inputs from Mid format (batch, heads, embed_dim, seq_len)
        # to standard format (batch, heads, seq_len, embed_dim)
        q_transposed = query.transpose(-2, -1)
        k_transposed = key.transpose(-2, -1)
        v_transposed = value.transpose(-2, -1)

        # Call base XLA implementation with mapped parameter names
        result = super().execute(
            q_transposed,
            k_transposed,
            v_transposed,
            is_causal=use_causal_mask,
            dropout_p=dropout_p,
        )

        if not result.success:
            return result

        # Output is already in correct format (batch, heads, seq_len, embed_dim)
        if out is not None:
            out.copy_(result.output)
            return ExecutionResult(success=True, output=out)

        return result
