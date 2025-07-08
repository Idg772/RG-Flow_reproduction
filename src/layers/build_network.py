from __future__ import annotations

from typing import Sequence, List

from .ResidualNetwork import ResidualNetwork    
from .RNVP            import RNVPBlock          
from .MERA            import MERABlock, RG_Flow 


def _normalise_block_spec(
    depth: int,
    spec: int | Sequence[int],
) -> List[int]:
    """
    Normalise *blocks_per_layer* into an explicit list.

    Returns
    -------
    list[int]  –  length == *depth*
    """
    if isinstance(spec, int):
        return [int(spec)] * depth

    if not isinstance(spec, (list, tuple)):
        raise TypeError("blocks_per_layer must be int or sequence of int")

    if len(spec) != depth:
        raise ValueError(
            f"blocks_per_layer has length {len(spec)} but depth is {depth}"
        )

    return [int(x) for x in spec]


# ----------------------------------------------------------------
def build_rg_flow(
    *,
    depth: int = 8,
    blocks_per_layer: int | Sequence[int] = 4,
    kernel_size: int = 4,
    num_channels: int = 3,
    apply_tanh: bool = True,
    use_ckpt: bool = True,
) -> RG_Flow:
    """
    Construct a full RG‑Flow (MERA) model.

    Parameters
    ----------
    depth : int
        Number of MERA levels (→ total RG steps).
    blocks_per_layer : int | list[int]
        • **int**    same count for every level (backwards‑compatible)  
        • **list**   explicit per‑level counts,
          e.g. ``[8, 8, 6, 6, 4, 4, 2, 2]`` for depth = 8.
    kernel_size : int
        Pixel‑block size *m* processed by every RNVP coupling layer.
    num_channels : int
        Image channels (3 for RGB).
    apply_tanh / use_ckpt : bool
        Passed straight through to `RNVPBlock`.
    """
    n_blocks = _normalise_block_spec(depth, blocks_per_layer)

    mera_layers = []
    for level, n in enumerate(n_blocks):
        # -- build one RNVP coupling layer ----------------------------------
        s_nets = [ResidualNetwork() for _ in range(n)]
        t_nets = [ResidualNetwork() for _ in range(n)]
        rnvp   = RNVPBlock(
            kernel_size  = kernel_size,
            num_channels = num_channels,
            s_nets       = s_nets,
            t_nets       = t_nets,
            apply_tanh   = apply_tanh,
            use_ckpt     = use_ckpt,
        )

        # -- wrap into a MERABlock ------------------------------------------
        mera_layers.append(
            MERABlock(
                m       = kernel_size,
                h       = level // 2,          # dilation exponent
                network = rnvp,
                shift   = (level % 2 == 1),    # alternate shifting pattern
            )
        )

    return RG_Flow(mera_layers)


# ----------------------------------------------------------------
# optional sanity‑check when executed directly
if __name__ == "__main__":  # pragma: no cover
    model = build_rg_flow(depth=2, blocks_per_layer=[2, 1])
    print(f"✔ RG‑Flow built: {sum(p.numel() for p in model.parameters()):,} parameters")
