import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------- autograd helper -----------------
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

# ----------------- physics loss -----------------
def physics_loss(model: nn.Module, R, Tenv, t0=0.0, t1=1000.0, n=1000):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    ts = torch.linspace(t0, t1, steps=n, device=device, dtype=dtype).view(-1, 1)
    ts.requires_grad_(True)

    R = torch.as_tensor(R, device=device, dtype=dtype)
    Tenv = torch.as_tensor(Tenv, device=device, dtype=dtype)

    T = model(ts)
    dTdt = grad(T, ts)

    residual = dTdt - R * (Tenv - T)   # should be 0
    return (residual ** 2).mean()

# ----------------- initial condition loss -----------------
def ic_loss(model: nn.Module, t_ic, T0):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    t_ic = torch.tensor([[t_ic]], device=device, dtype=dtype)
    T0 = torch.tensor([[T0]], device=device, dtype=dtype)
    return ((model(t_ic) - T0) ** 2).mean()

# ----------------- simple PINN model -----------------
class MLP(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------- train + plot -----------------
def main():
    # Given parameters
    R = 0.05
    Tenv = 25.0

    # Choose an initial condition (EDIT THIS)
    T0 = 100.0
    t_ic = 0.0

    device = "cpu"
    model = MLP(width=64, depth=3).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for step in range(5000):
        opt.zero_grad()
        loss_pde = physics_loss(model, R=R, Tenv=Tenv, t0=0.0, t1=200.0, n=800)
        loss_ic  = ic_loss(model, t_ic=t_ic, T0=T0)
        loss = loss_pde + 100.0 * loss_ic  # IC weight matters
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(f"step {step:5d} | loss {loss.item():.3e} | pde {loss_pde.item():.3e} | ic {loss_ic.item():.3e}")

    # Plot PINN vs analytic
    t = torch.linspace(0.0, 200.0, steps=500).view(-1, 1)
    with torch.no_grad():
        T_pinn = model(t).cpu().numpy()

    t_np = t.cpu().numpy()
    T_true = Tenv + (T0 - Tenv) * torch.exp(-torch.tensor(R) * t).cpu().numpy()

    plt.figure()
    plt.plot(t_np, T_pinn, label="PINN")
    plt.plot(t_np, T_true, "--", label="Analytic")
    plt.xlabel("t")
    plt.ylabel("T(t)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()