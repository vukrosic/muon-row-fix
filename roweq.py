"""Muon row fix (roweq): after Newton-Schulz, give every row of a tall update the same length.

Same as Aurora's U-NorMuon (arXiv 2606.27715). Wide and square matrices already have equal
rows after Muon, so they pass through unchanged.
"""
import torch


def roweq(O: torch.Tensor) -> torch.Tensor:
    """O: Muon's orthogonalized update (rows = output neurons). Keeps the total (Frobenius) size."""
    if O.shape[-2] > O.shape[-1]:
        O = O / O.norm(dim=-1, keepdim=True).clamp_min(1e-12) * O.norm(dim=(-2, -1), keepdim=True) / O.shape[-2] ** 0.5
    return O


# Where it goes in a typical Muon step:
#   O = newton_schulz(G)          # existing Muon call
#   O = roweq(O)                  # add this line
#   W -= lr * scale * O           # existing update

if __name__ == "__main__":
    torch.manual_seed(0)
    from torch.linalg import svd

    def polar(G):  # exact orthogonalization, stands in for Newton-Schulz
        U, _, Vh = svd(G, full_matrices=False)
        return U @ Vh

    tall = polar(torch.randn(512, 128) * torch.rand(512, 1) * 3)
    wide = polar(torch.randn(128, 512))
    rt = tall.norm(dim=1)
    print(f"tall before: row lengths {rt.min():.3f}..{rt.max():.3f}")
    ft = roweq(tall)
    r = ft.norm(dim=1)
    assert torch.allclose(r, r[0].expand_as(r), atol=1e-5), "rows not equal"
    assert torch.allclose(ft.norm(), tall.norm(), atol=1e-4), "total size changed"
    assert torch.allclose((ft / ft.norm(dim=1, keepdim=True) * tall.norm(dim=1, keepdim=True)), tall, atol=1e-5), "direction changed"
    print(f"tall after:  row lengths {r.min():.3f}..{r.max():.3f}, total {ft.norm():.3f} = {tall.norm():.3f}")
    assert torch.equal(roweq(wide), wide), "wide changed"
    b = roweq(torch.stack([tall, tall]))
    assert torch.allclose(b[1], ft, atol=1e-6), "batched differs"
    print("wide: unchanged. batched: same. all checks pass.")
