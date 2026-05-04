"""
Bilevel optimisation for mixture-of-exponentials fitting under Rician noise.

Problem:
  True signal:  G(x*)[i] = c0*exp(-lam0*t_i) + c1*exp(-lam1*t_i)
  Observed:     y* ~ Rician(G(x*), sigma)  or  y* = G(x*) + Gaussian noise

  Upper level:  min_{mu >= 0}  V(mu) = ||y* - G(xhat(mu))||^2
  Lower level:  xhat(mu) = argmin_x  ||y* - G(x)||^2 + mu*||x||^2
                            s.t.  c >= 0,  lambda > 0   (via softplus reparametrisation)

Backward of XhatModule uses implicit differentiation:
    d xhat / d mu = -H^{-1} xhat
where H = J_G^T J_G + mu*I  (Gauss-Newton approximation).

Two outer solvers compared:
  (A) Gradient descent  on log_mu
  (B) Gauss-Newton      on mu  (scalar, with line search)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
from tqdm import tqdm
# ─────────────────────────────────────────────────────────────────────────────
# 1.  Signal model  G(c, lam, t)
# ─────────────────────────────────────────────────────────────────────────────

def biexponential(t, c, lam):
    return c[0]*torch.exp(-lam[0]*t) + c[1]*torch.exp(-lam[1]*t)

def jacobian_biexp(t, c, lam):
    """Returns J in R^{n x 4}, cols = [dG/dc0, dG/dc1, dG/dlam0, dG/dlam1]"""
    e0 = torch.exp(-lam[0]*t)
    e1 = torch.exp(-lam[1]*t)
    return torch.stack([e0, e1, -c[0]*t*e0, -c[1]*t*e1], dim=1)  # (n,4)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Noise models
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(signal, sigma):
    return signal + sigma*torch.randn_like(signal)

def add_rician_noise(signal, sigma):
    nr = sigma*torch.randn_like(signal)
    ni = sigma*torch.randn_like(signal)
    return torch.sqrt((signal+nr)**2 + ni**2)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Reparametrisation helpers  (ensures c>0, lam>0)
# ─────────────────────────────────────────────────────────────────────────────

def sp(u):   return torch.log1p(torch.exp(u))
def sp_d(u): return torch.sigmoid(u)
def inv_sp(x): return torch.log(torch.exp(x.clamp(min=1e-6)) - 1 + 1e-8)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Inner solver: Gauss-Newton for the lower-level NLLS
# ─────────────────────────────────────────────────────────────────────────────

def lower_level_gn_ep(y, mu, t, beta=0.0, x_star=None, max_iter=300, tol=1e-9, x_init=None):
    """
    min_u  ||y - G(softplus(u))||^2 + (mu + beta)*||softplus(u) - x_prior||^2
    where x_prior = (beta / (mu + beta)) * x_star.
    
    Returns x = softplus(u) and loss history.
    """
    if x_init is None:
        x_init = torch.tensor([0.6, 0.4, 0.5, 2.0])
    u = inv_sp(x_init.clamp(min=1e-3)).clone().detach()

    # 1. Setup effective parameters based on the EP factorization
    mu_eff = mu + beta
    if beta > 0.0 and x_star is not None:
        x_prior = (beta / mu_eff) * x_star
    else:
        x_prior = torch.zeros_like(x_init)

    hist = []
    for _ in range(max_iter):
        x   = sp(u)
        c, lam = x[:2], x[2:]
        Gx  = biexponential(t, c, lam)
        r   = y - Gx
        
        # 2. Refactorized loss
        loss = (r**2).sum() + mu_eff * ((x - x_prior)**2).sum()
        hist.append(loss.item())

        Jx = jacobian_biexp(t, c, lam)           # (n,4)
        sd = sp_d(u)                             # (4,)
        Ju = Jx * sd.unsqueeze(0)                # (n,4)  chain rule

        # 3. Refactorized Gauss-Newton Hessian and Gradient
        H_gn = Ju.T @ Ju + mu_eff * torch.eye(4) + 1e-7 * torch.eye(4)
        grad = -Ju.T @ r + mu_eff * (x - x_prior) * sd
        
        du   = torch.linalg.solve(H_gn, -grad)

        # 4. Refactorized Armijo line search
        step = 1.0
        for _ in range(25):
            u_new = u + step * du
            x_new = sp(u_new)
            r_new = y - biexponential(t, x_new[:2], x_new[2:])
            
            loss_new = (r_new**2).sum() + mu_eff * ((x_new - x_prior)**2).sum()
            
            if loss_new <= loss - 1e-4 * step * (grad * du).sum():
                break
            step *= 0.5

        u = (u + step * du).detach()
        if step * du.norm() < tol:
            break

    return sp(u).detach(), hist

# ─────────────────────────────────────────────────────────────────────────────
# 5.  XhatModule — differentiable w.r.t. mu via implicit differentiation
# ─────────────────────────────────────────────────────────────────────────────

class _XhatFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_scalar, y, t, x_init):
        mu_val = mu_scalar.item()
        x_sol, _ = lower_level_gn_ep(y, mu_val, t, x_init=x_init, \
                                     beta = 0.0, x_star=torch.zeros_like(x_init))
        c, lam = x_sol[:2], x_sol[2:]
        Jx = jacobian_biexp(t, c, lam)
        H  = Jx.T @ Jx + mu_val*torch.eye(4) + 1e-7*torch.eye(4)
        ctx.save_for_backward(x_sol, H)
        return x_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        d xhat / d mu = -H^{-1} xhat   (implicit differentiation)
        dL/d mu = (dL/d xhat)^T (d xhat/d mu) = -(dL/d xhat)^T H^{-1} xhat
        """
        x_sol, H = ctx.saved_tensors
        z   = torch.linalg.solve(H, x_sol)        # H^{-1} xhat
        dmu = -(grad_output * z).sum().reshape(1)
        return dmu, None, None, None


class XhatModule(nn.Module):
    """
    Learnable log_mu; forward returns xhat(mu) with implicit-diff backward.
    """
    def __init__(self, mu_init=1.0):
        super().__init__()
        self.log_mu = nn.Parameter(torch.tensor(float(np.log(mu_init))))

    @property
    def mu(self):
        return torch.exp(self.log_mu)

    def forward(self, y, t, x_init=None):
        if x_init is None:
            x_init = torch.tensor([0.6, 0.4, 0.5, 2.0])
        return _XhatFn.apply(self.mu, y, t, x_init)

# ─────────────────────────────────────────────────────────────────────────────
# 6a.  Gradient descent on mu  (outer optimiser A)
# ─────────────────────────────────────────────────────────────────────────────

def gradient_descent_mu(y, t, mu_init=1.0, lr=0.15, n_steps=100):
    """
    Uses XhatModule + manual gradient computation via implicit diff.
    Optimises log_mu with plain SGD step.
    """
    log_mu = torch.tensor(float(np.log(mu_init)), requires_grad=False)
    x_init = torch.tensor([0.6, 0.4, 0.5, 2.0])
    hist   = {"mu": [], "V": [], "xhat": []}

    for _ in tqdm(range(n_steps), desc="outer loop step"):
        mu_val = torch.exp(log_mu).item()
        xhat, _ = lower_level_gn_ep(y, mu_val, t, x_init=x_init, \
                                    beta = 0.0, x_star=torch.zeros_like(x_init))
        c, lam  = xhat[:2], xhat[2:]
        Gx = biexponential(t, c, lam)
        r  = y - Gx
        Jx = jacobian_biexp(t, c, lam)
        H  = Jx.T @ Jx + mu_val*torch.eye(4) + 1e-7*torch.eye(4)

        # sensitivity: j = J_G H^{-1} xhat  ∈ R^n
        z  = torch.linalg.solve(H, xhat)
        j  = Jx @ z

        # dV/dmu   = 2 r^T j
        # dV/d(log mu) = dV/dmu * mu
        dV_dmu     = 2*(r*j).sum()
        dV_dlogmu  = dV_dmu * torch.exp(log_mu)

        log_mu = (log_mu - lr*dV_dlogmu).detach()

        V = (r**2).sum().item()
        hist["mu"].append(torch.exp(log_mu).item())
        hist["V"].append(V)
        hist["xhat"].append(xhat)
        x_init = xhat   # warm start

    return hist

# ─────────────────────────────────────────────────────────────────────────────
# 6b.  Gauss-Newton on mu  (outer optimiser B)
# ─────────────────────────────────────────────────────────────────────────────

def gauss_newton_mu(y, t, mu_init=1.0, n_steps=100):
    """
    Scalar GN step: mu <- mu - (j^T r)/(j^T j)
    where j = J_G H^{-1} xhat is the sensitivity of the residual to mu.
    """
    mu     = torch.tensor(float(mu_init))
    x_init = torch.tensor([0.6, 0.4, 0.5, 2.0])
    hist   = {"mu": [], "V": [], "xhat": []}

    for _ in tqdm(range(n_steps), desc="outer loop step"):
        mu_val = mu.item()
        xhat, _ = lower_level_gn_ep(y, mu_val, t, x_init=x_init, \
                                    beta = 0.0, x_star=torch.zeros_like(x_init))
        c, lam  = xhat[:2], xhat[2:]
        Gx = biexponential(t, c, lam)
        r  = y - Gx
        Jx = jacobian_biexp(t, c, lam)
        H  = Jx.T @ Jx + mu_val*torch.eye(4) + 1e-7*torch.eye(4)

        z      = torch.linalg.solve(H, xhat)
        j      = Jx @ z                          # (n,)
        delta  = (j*r).sum() / ((j*j).sum() + 1e-10)

        # line search: keep mu > 0
        step = 1.0
        for _ in range(20):
            mu_new = mu - step*delta
            if mu_new.item() > 1e-6:
                break
            step *= 0.5
        mu = (mu - step*delta).detach().clamp(min=1e-6)

        hist["mu"].append(mu.item())
        hist["V"].append((r**2).sum().item())
        hist["xhat"].append(xhat)
        x_init = xhat

    return hist

# ─────────────────────────────────────────────────────────────────────────────
# 6C.  Equilibrium propagation
# ─────────────────────────────────────────────────────────────────────────────

def ep_gradient_descent_mu(y, t, x_star, mu_init=1.0, beta=0.01, lr=0.1, n_steps=100):
    """
    Outer loop updating mu via Equilibrium Propagation gradient estimate.
    Matches the hist structure of gauss_newton_mu for easy downstream comparison.
    """
    mu = torch.tensor(float(mu_init))
    x_init = torch.tensor([0.6, 0.4, 0.5, 2.0])
    hist = {"mu": [], "V": [], "xhat": []}

    for _ in tqdm(range(n_steps), desc="outer loop step"):
        mu_val = mu.item()
        
        # 1. Phase 1: Free state (standard Tikhonov)
        x0, _ = lower_level_gn_ep(y, mu_val, t, beta=0.0, x_star=x_star, x_init=x_init)
        
        # 2. Phase 2: Nudged state (shifted Tikhonov)
        # Warm start using x0 for extremely fast convergence
        x_beta, _ = lower_level_gn_ep(y, mu_val, t, beta=beta, x_star=x_star, x_init=x0)

        # 3. EP Gradient estimate for mu
        # dL/dmu ≈ (dE/dmu(x_beta) - dE/dmu(x0)) / beta
        # Since E contains 0.5 * mu * ||x||^2, dE/dmu = 0.5 * ||x||^2
        grad_mu = (0.5 * (x_beta**2).sum() - 0.5 * (x0**2).sum()) / beta

        # 4. Gradient descent step on mu with line search for positivity
        step = lr
        for _ in range(20):
            mu_new = mu - step * grad_mu
            if mu_new.item() > 1e-6:
                break
            step *= 0.5
        
        mu = (mu - step * grad_mu).detach().clamp(min=1e-6)

        # 5. Calculate V(mu) = ||y - G(x0)||^2 to match the GN history tracking
        c, lam = x0[:2], x0[2:]
        Gx = biexponential(t, c, lam)
        r = y - Gx
        V = (r**2).sum().item()

        hist["mu"].append(mu.item())
        hist["V"].append(V)
        hist["xhat"].append(x0)
        
        # Warm start next iteration
        x_init = x0

    return hist


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Synthetic experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(noise_type="gaussian", sigma=0.00):
    t       = torch.linspace(0.05, 3.0, 60)
    c_true  = torch.tensor([0.7, 0.3])
    lam_true= torch.tensor([0.4, 0.5])
    x_star  = torch.cat([c_true, lam_true])
    G_true  = biexponential(t, c_true, lam_true)

    y = add_gaussian_noise(G_true, sigma) if noise_type == "gaussian" \
        else add_rician_noise(G_true, sigma)

    mu_init    = 0.5
    gd_steps   = 500
    gn_steps   = 500
    ep_steps   = 500
    gd_lr      = 0.12
    ep_lr      = 0.02
    ep_beta    = 1e-5

    print(f"  [{noise_type}] running GD ...")
    hist_gd = gradient_descent_mu(y, t, mu_init=mu_init, lr=gd_lr, n_steps=gd_steps)
    print(f"  [{noise_type}] running GN ...")
    hist_gn = gauss_newton_mu(y, t, mu_init=mu_init, n_steps=gn_steps)
    print(f"  [{noise_type}] running EP ...")
    hist_ep = ep_gradient_descent_mu(y, t, x_star=x_star, mu_init=mu_init,
                                     beta=ep_beta, lr=ep_lr, n_steps=ep_steps)

    return t, G_true, y, hist_gd, hist_gn, hist_ep, c_true, lam_true

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_figure():
    BG    = "white"; PANEL = "white"; GRID = "#d0d0d0"
    C_GD  = "tab:red"; C_GN  = "tab:green"; C_TRUE = "black"
    C_EP  = "tab:orange"; C_OBS = "tab:gray"; ACCENT = "black"

    fig = plt.figure(figsize=(22, 12), facecolor=BG)
    fig.suptitle(
        "Bilevel Optimisation  ·  Mixture-of-Exponentials  ·  Three Outer Solvers\n",
        color=C_TRUE, fontsize=14, y=0.985)

    gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.52,
                                  top=0.94, bottom=0.06)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="black", labelsize=12)
        for sp in ax.spines.values():
            sp.set_edgecolor("black")
        ax.grid(color=GRID, lw=0.6, ls="--", alpha=0.7)
        ax.set_title(title, color=C_TRUE, fontsize=14, pad=6)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")

    for row, (noise_type, label) in enumerate([("gaussian","Gaussian noise"),
                                                ("rician",  "Rician noise")]):
        gs = gridspec.GridSpecFromSubplotSpec(1, 4,
            subplot_spec=gs_outer[row], wspace=0.34)
        t, G_true, y, h_gd, h_gn, h_ep, c_true, lam_true = \
            run_experiment(noise_type, sigma=0.05)
        x_star = torch.cat([c_true, lam_true])

        mu_gd = h_gd["mu"][-1];  mu_gn = h_gn["mu"][-1]; mu_ep = h_ep["mu"][-1]
        xh_gd = h_gd["xhat"][-1]
        xh_gn = h_gn["xhat"][-1]
        xh_ep = h_ep["xhat"][-1]
        Gh_gd = biexponential(t, xh_gd[:2], xh_gd[2:]).detach()
        Gh_gn = biexponential(t, xh_gn[:2], xh_gn[2:]).detach()
        Gh_ep = biexponential(t, xh_ep[:2], xh_ep[2:]).detach()

        rmse_gd = [torch.sqrt(torch.mean((xhat - x_star)**2)).item() for xhat in h_gd["xhat"]]
        rmse_gn = [torch.sqrt(torch.mean((xhat - x_star)**2)).item() for xhat in h_gn["xhat"]]
        rmse_ep = [torch.sqrt(torch.mean((xhat - x_star)**2)).item() for xhat in h_ep["xhat"]]

        # panel 1 – signal fit
        ax1 = fig.add_subplot(gs[0])
        style(ax1, f"[{label}]  Signal fit")
        ax1.scatter(t.numpy(), y.numpy(), s=7, color=C_OBS,
                    alpha=0.55, label="observations $y^*$", zorder=2)
        ax1.plot(t.numpy(), G_true.numpy(), color=C_TRUE,
                 lw=1.8, label="true $G(x^*)$", zorder=4)
        ax1.plot(t.numpy(), Gh_gd.numpy(), color=C_GD,
                 lw=1.5, ls="--", label=f"GD  $\\mu$={mu_gd:.3f}", zorder=5)
        ax1.plot(t.numpy(), Gh_gn.numpy(), color=C_GN,
                 lw=1.5, ls="-.", label=f"GN  $\\mu$={mu_gn:.3f}", zorder=5)
        ax1.plot(t.numpy(), Gh_ep.numpy(), color=C_EP,
             lw=1.5, ls=":", label=f"EP  $\\mu$={mu_ep:.3f}", zorder=5)
        ax1.set_xlabel("time $t$", fontsize=14)
        ax1.set_ylabel("signal", fontsize=14)
        ax1.legend(fontsize=10, framealpha=0.9)

        # panel 2 – mu convergence
        ax2 = fig.add_subplot(gs[1])
        style(ax2, f"[{label}]  $\\mu$ path")
        ax2.semilogy(h_gd["mu"], color=C_GD, lw=1.8, label="Grad. descent")
        ax2.semilogy(h_gn["mu"], color=C_GN, lw=1.8, ls="--", label="Gauss-Newton")
        ax2.semilogy(h_ep["mu"], color=C_EP, lw=1.8, ls=":", label="Eq. propagation")
        ax2.set_xlabel("outer iteration", fontsize=14)
        ax2.set_ylabel("$\\mu$  (log scale)", fontsize=14)
        ax2.legend(fontsize=10, framealpha=0.9)

        # panel 3 – V(mu) convergence
        ax3 = fig.add_subplot(gs[2])
        style(ax3, f"[{label}]  Upper-level loss $V(\\mu)$")
        ax3.semilogy(h_gd["V"], color=C_GD, lw=1.8, label="Grad. descent")
        ax3.semilogy(h_gn["V"], color=C_GN, lw=1.8, ls="--", label="Gauss-Newton")
        ax3.semilogy(h_ep["V"], color=C_EP, lw=1.8, ls=":", label="Eq. propagation")
        ax3.set_xlabel("outer iteration", fontsize=14)
        ax3.set_ylabel("$\\|y^*-G(\\hat{x})\\|^2$  (log)", fontsize=14)
        ax3.legend(fontsize=10, framealpha=0.9)

        # panel 4 - solution error RMSE
        ax4 = fig.add_subplot(gs[3])
        style(ax4, f"[{label}]  Solution error RMSE")
        ax4.semilogy(rmse_gd, color=C_GD, lw=1.8, label="Grad. descent")
        ax4.semilogy(rmse_gn, color=C_GN, lw=1.8, ls="--", label="Gauss-Newton")
        ax4.semilogy(rmse_ep, color=C_EP, lw=1.8, ls=":", label="Eq. propagation")
        ax4.set_xlabel("outer iteration", fontsize=14)
        ax4.set_ylabel(r"$\mathrm{RMSE}(\hat{x}, x^*)$", fontsize=14)
        ax4.legend(fontsize=10, framealpha=0.9)

        fig.text(0.005, 0.75 - row*0.475, label.upper(),
             color=ACCENT, fontsize=10,
                 rotation=90, va="center")

    plt.savefig("rician_bilevel.png",
            dpi=155, bbox_inches="tight")
    print("Figure saved.")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_figure()
