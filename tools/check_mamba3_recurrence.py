"""Does the community reference implement the paper's trapezoidal recurrence?

Paper (Mamba-3, exponential-trapezoidal):
    h_t = a_t h_{t-1} + (1-L_t) dt_t a_t (B_{t-1} x_{t-1}) + L_t dt_t (B_t x_t)
    with a_t = exp(dt_t A_t)      <-- the PREVIOUS term carries the decay a_t

rishikksh20/mamba3-pytorch `mamba3_siso_scan`:
    blended = (1-tr) Bx_t + tr*0.5*(Bx_t + Bx_{t-1})
    h_t     = a_t h_{t-1} + dt_t * blended
            = a_t h_{t-1} + dt_t (1 - tr/2)(B_t x_t) + dt_t (tr/2)(B_{t-1}x_{t-1})
                                                       ^^^^ no a_t factor

If these differ, the community repo cannot serve as ground truth for a kernel
meant to reproduce the official checkpoints.
"""
import torch

torch.manual_seed(0)
Bb, L, H, P, D = 1, 6, 2, 3, 4

x = torch.randn(Bb, L, H, P)
Bp = torch.randn(Bb, L, H, D)
dt = torch.rand(Bb, L, H) * 0.5 + 0.05
A = -(torch.rand(Bb, L, H) * 2 + 0.5)
adt = A * dt
gate = torch.rand(Bb, L, H)          # lambda / trap, in [0,1]


def ref_community(gate):
    """Exactly the arithmetic in mamba3-pytorch."""
    h = torch.zeros(Bb, H, P, D)
    Bx_prev = torch.zeros(Bb, H, P, D)
    out = []
    for t in range(L):
        decay = torch.exp(adt[:, t]).unsqueeze(-1).unsqueeze(-1)
        Bx = torch.einsum("bhp,bhd->bhpd", x[:, t], Bp[:, t])
        tr = gate[:, t].unsqueeze(-1).unsqueeze(-1)
        d = dt[:, t].unsqueeze(-1).unsqueeze(-1)
        blended = (1 - tr) * Bx + tr * 0.5 * (Bx + Bx_prev)
        h = decay * h + d * blended
        out.append(h.clone())
        Bx_prev = Bx
    return torch.stack(out, 1)


def ref_paper(gate):
    """h_t = a h_{t-1} + (1-L) dt a Bx_{t-1} + L dt Bx_t."""
    h = torch.zeros(Bb, H, P, D)
    Bx_prev = torch.zeros(Bb, H, P, D)
    out = []
    for t in range(L):
        a = torch.exp(adt[:, t]).unsqueeze(-1).unsqueeze(-1)
        Bx = torch.einsum("bhp,bhd->bhpd", x[:, t], Bp[:, t])
        lam = gate[:, t].unsqueeze(-1).unsqueeze(-1)
        d = dt[:, t].unsqueeze(-1).unsqueeze(-1)
        h = a * h + (1 - lam) * d * a * Bx_prev + lam * d * Bx
        out.append(h.clone())
        Bx_prev = Bx
    return torch.stack(out, 1)


c, p = ref_community(gate), ref_paper(gate)
print(f"community vs paper, same gate : max_abs {float((c-p).abs().max()):.4e}")

# Could the gate just be parameterised differently? Try every remapping that
# would make them agree if the ONLY difference were the gate convention.
for name, g in [("1-gate", 1 - gate), ("gate/2", gate / 2),
                ("1-gate/2", 1 - gate / 2), ("2*gate", (2 * gate).clamp(max=1))]:
    d = float((ref_community(gate) - ref_paper(g)).abs().max())
    print(f"community(gate) vs paper({name:9s}): max_abs {d:.4e}")

# Isolate the structural difference: does the previous term carry the decay?
lam = gate
h1 = h2 = torch.zeros(Bb, H, P, D)
Bx_prev = torch.zeros(Bb, H, P, D)
worst = 0.0
for t in range(L):
    a = torch.exp(adt[:, t]).unsqueeze(-1).unsqueeze(-1)
    Bx = torch.einsum("bhp,bhd->bhpd", x[:, t], Bp[:, t])
    l = lam[:, t].unsqueeze(-1).unsqueeze(-1)
    d = dt[:, t].unsqueeze(-1).unsqueeze(-1)
    h1 = a * h1 + (1 - l) * d * a * Bx_prev + l * d * Bx      # with decay
    h2 = a * h2 + (1 - l) * d * Bx_prev + l * d * Bx          # without
    worst = max(worst, float((h1 - h2).abs().max()))
    Bx_prev = Bx
print(f"\nsame gate, ONLY difference = decay on the previous term: "
      f"max_abs {worst:.4e}")
print("(if this is non-zero, the two recurrences are genuinely different)")
