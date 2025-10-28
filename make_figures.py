# make_figures.py  — QIP-ready figures (symbols & labels unified)
import os, argparse
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True)
    return path

def savefig(fig, filename):
    ensure_dir(os.path.dirname(filename) or ".")
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def time_grid(tmax=20.0, num=2001):
    return np.linspace(0.0, tmax, num=num)

def H2(x):
    x = np.clip(x, 1e-15, 1-1e-15)
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

# ---------- Markovian Pauli / SU(3) ----------
def lambda_pauli(t, mu, Gamma):
    return (1.0 - mu)*np.exp(-2.0*Gamma*t) + mu*np.exp(-Gamma*t)

def lambda_qutrit_SU3(t, mu, Gamma):
    lam3 = np.exp(-Gamma*t)
    return (1.0 - mu)*(lam3**2) + mu*lam3

# ---------- GAD (single-site) ----------
def gamma_markov(t, Gamma):
    return 1.0 - np.exp(-Gamma*t)

def lambda_GAD_from_gamma(mu, gamma):
    # two-site convex law (used for >=2 sites);
    # for single-qubit C_{l1} we will use sqrt(1-gamma) directly.
    return (1.0 - mu)*(1.0 - gamma) + mu*np.sqrt(1.0 - gamma)

# ---------- JC amplitude (non-Markovian) ----------
def G_JC(t, lam, gamma0=1.0):
    d2 = lam**2 - 2.0*gamma0*lam
    d = np.sqrt(d2 + 0j)
    return np.exp(-lam*t/2.0) * (np.cosh(d*t/2.0) + (lam/d)*np.sinh(d*t/2.0))

# ---------- PMME amplitude (non-Markovian) ----------
# Paper-consistent: F(t) = e^{-α t/2} [ cosh(Δ t/2) + (α/Δ) sinh(Δ t/2) ],
# Δ = sqrt(α^2 - 2 α γ_M)
def F_PMME(t, alpha, gammaM=1.0):
    Delta = np.sqrt(alpha**2 - 2.0*alpha*gammaM + 0j)
    return np.exp(-alpha*t/2.0) * (np.cosh(Delta*t/2.0) + (alpha/Delta)*np.sinh(Delta*t/2.0))

def lambda_mixture_from_amp(mu, A_abs):
    return (1.0 - mu)*(A_abs**2) + mu*A_abs

# =========================
#      FIGURE MAKERS
# =========================

# --- Pauli (qubits) ---
def fig3_1a_pauli2q_compare(Gamma=0.2, mu=0.6, tmax=25.0):
    t = time_grid(tmax)
    lam = lambda_pauli(t, mu, Gamma)
    # FIX: two qubits => (2^2 - 1) * lam^2 = 3 * lam^2
    C_analytic = 3.0 * (lam**2)
    tidx = np.linspace(0, len(t)-1, 60, dtype=int)
    t_dots = t[tidx]; C_dots = C_analytic[tidx]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, C_analytic, label="analytic")
    ax.plot(t_dots, C_dots, linestyle="None", marker="o", markersize=3, label="numerical")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (two qubits)")
    ax.set_title(r"$\mu=%.1f$, $\Gamma=%.2f$"%(mu, Gamma))
    ax.legend()
    savefig(fig, "figures/fig3_1a_pauli2q_compare_2.pdf")

def fig3_2_pauli_nq(Gamma=0.2, mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0):
    t = time_grid(tmax)
    lam = lambda_pauli(t, mu, Gamma)
    fig, ax = plt.subplots(figsize=(6,4))
    for n in n_list:
        Cn = (2**n - 1.0) * (lam**n)
        ax.plot(t, Cn, label=f"n={n}")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$")
    ax.set_yscale("log")
    ax.set_title(r"Pauli, $\mu=%.1f$, $\Gamma=%.2f$"%(mu,Gamma))
    ax.legend(ncol=2)
    savefig(fig, "figures/fig3_2_pauli_nq_2.pdf")

# --- SU(3) depolarizing (qutrits) ---
def fig3_3a_pauli1t_qutrit(Gamma=0.2, mu=0.6, tmax=25.0):
    t = time_grid(tmax)
    lam_tot = lambda_qutrit_SU3(t, mu, Gamma)
    C = 2.0 * lam_tot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, C)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (single qutrit)")
    ax.set_title(r"SU(3) depolarizing, $\mu=%.1f$, $\Gamma=%.2f$"%(mu,Gamma))
    savefig(fig, "figures/fig3_3a_pauli1t_compare_2.pdf")

def fig3_4_pauli_nt_qutrit(Gamma=0.2, mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0):
    t = time_grid(tmax)
    lam_tot = lambda_qutrit_SU3(t, mu, Gamma)
    fig, ax = plt.subplots(figsize=(6,4))
    for n in n_list:
        Cn = (3**n - 1.0) * (lam_tot**n)
        ax.plot(t, Cn, label=f"n={n}")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$")
    ax.set_yscale("log")
    ax.set_title(r"SU(3) depolarizing, $\mu=%.1f$, $\Gamma=%.2f$"%(mu,Gamma))
    ax.legend(ncol=2)
    savefig(fig, "figures/fig3_4_pauli_nt_2.pdf")

# --- GAD (qubits/qutrits) ---
def fig4_1_GAD_2q(mu_list=(0.0,0.3,0.6,1.0), tmax=25.0):
    t = time_grid(tmax)
    gamma = 1.0 - np.exp(-0.2*t)
    fig, ax = plt.subplots(figsize=(6,4))
    for mu in mu_list:
        lam = lambda_GAD_from_gamma(mu, gamma)
        # FIX: two qubits => 3 * lam^2
        C = 3.0 * (lam**2)
        ax.plot(t, C, label=r"$\mu={}$".format(mu))
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (two qubits)")
    ax.set_title(r"GAD: $\gamma(t)=1-e^{-0.2 t}$")
    ax.legend()
    savefig(fig, "figures/fig4_1_GAD_2q.pdf")

def fig4_2_GAD_nq(mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0):
    t = time_grid(tmax)
    gamma = 1.0 - np.exp(-0.2*t)
    lam = lambda_GAD_from_gamma(mu, gamma)
    fig, ax = plt.subplots(figsize=(6,4))
    for n in n_list:
        Cn = (2**n - 1.0) * (lam**n)
        ax.plot(t, Cn, label=f"n={n}")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$")
    ax.set_yscale("log")
    ax.set_title(r"GAD: $\mu=%.1f$"%mu)
    ax.legend(ncol=2)
    savefig(fig, "figures/fig4_2_GAD_nq.pdf")

def fig4_3_GAD_nt(mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0):
    t = time_grid(tmax)
    gamma = 1.0 - np.exp(-0.2*t)
    lam = lambda_GAD_from_gamma(mu, gamma)
    fig, ax = plt.subplots(figsize=(6,4))
    for n in n_list:
        Cn = (3**n - 1.0) * (lam**n)
        ax.plot(t, Cn, label=f"n={n}")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$")
    ax.set_yscale("log")
    ax.set_title(r"Qutrit GAD: $\mu=%.1f$"%mu)
    ax.legend(ncol=2)
    savefig(fig, "figures/fig4_3_GAD_nt.pdf")

def fig4_4_GAD_temp_single_qubit(gamma_const=0.3, num=501):
    # single qubit: C_{l1} = sqrt(1 - gamma), independent of q and μ
    q_vals = np.linspace(0.0, 1.0, num=num)
    gamma = gamma_const
    Delta = 0.5*gamma*(2.0*q_vals - 1.0)
    R = np.sqrt(Delta**2 + 0.25*(1.0 - gamma))
    # C_r
    x1 = np.clip(0.5 + Delta, 1e-15, 1-1e-15)
    x2 = np.clip(0.5 + R,     1e-15, 1-1e-15)
    Cr = (-x1*np.log2(x1) - (1-x1)*np.log2(1-x1)) - (-x2*np.log2(x2) - (1-x2)*np.log2(1-x2))
    # C_{l1} flat in q
    Cl1 = np.sqrt(1.0 - gamma)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(q_vals, np.full_like(q_vals, Cl1), label=r"$C_{\ell_1}$")
    ax.plot(q_vals, Cr, linestyle="--", label=r"$C_r$")
    ax.set_xlabel(r"excitation probability $q(T)$")
    ax.set_ylabel("coherence")
    ax.set_title(r"Single qubit: $\gamma=%.2f$ ($\mu$ irrelevant)"%(gamma_const,))
    ax.legend()
    savefig(fig, "figures/fig4_4_GAD_temp.pdf")

# --- JC family ---
def fig5_1a_JC_2q(mu_list=(0.0,0.3,0.6,1.0), lam=0.3, gamma0=1.0, tmax=20.0):
    t = time_grid(tmax)
    A = np.abs(G_JC(t, lam, gamma0))
    fig, ax = plt.subplots(figsize=(6,4))
    for mu in mu_list:
        lam_mix = (1.0 - mu)*(A**2) + mu*A
        # FIX: two qubits => 3 * lam_mix^2
        C = 3.0 * (lam_mix**2)
        ax.plot(t, C, label=r"$\mu={}$".format(mu))
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (two qubits)")
    ax.set_title(r"JC: $\lambda=%.2f$, $\gamma_0=%.1f$"%(lam, gamma0))
    ax.legend()
    savefig(fig, "figures/fig5_1a_JC_2q_3.pdf")

def fig5_2_JC_nq(mu=0.6, lam=0.3, gamma0=1.0, n_list=(1,2,4,6,8,10), tmax=20.0):
    t = time_grid(tmax)
    A = np.abs(G_JC(t, lam, gamma0))
    lam_mix = (1.0 - mu)*(A**2) + mu*A
    fig, ax = plt.subplots(figsize=(6,4))
    for n in n_list:
        Cn = (2**n - 1.0) * (lam_mix**n)
        ax.plot(t, Cn, label=f"n={n}")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$")
    ax.set_yscale("log")
    ax.set_title(r"JC: $\mu=%.1f$, $\lambda=%.2f$, $\gamma_0=%.1f$"%(mu,lam,gamma0))
    ax.legend(ncol=2)
    savefig(fig, "figures/fig5_2_JC_nq_2.pdf")

def fig5_3_JC_1t(mu=0.6, lam=0.3, gamma0=1.0, tmax=20.0):
    t = time_grid(tmax)
    A = np.abs(G_JC(t, lam, gamma0))
    lam3 = (1.0 - mu)*(A**2) + mu*A
    C = 2.0 * lam3
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, C)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (single qutrit)")
    ax.set_title(r"JC: $\mu=%.1f$, $\lambda=%.2f$, $\gamma_0=%.1f$"%(mu,lam,gamma0))
    savefig(fig, "figures/fig5_3_JC_1t_2.pdf")

# --- PMME family (paper-consistent symbols) ---
def fig5_5_PMME_2q(mu_list=(0.0,0.3,0.6,1.0), alpha=0.15, gammaM=1.0, tmax=20.0):
    t = time_grid(tmax)
    A = np.abs(F_PMME(t, alpha, gammaM))
    fig, ax = plt.subplots(figsize=(6,4))
    for mu in mu_list:
        lam_mix = (1.0 - mu)*(A**2) + mu*A
        # FIX: two qubits => 3 * lam_mix^2
        C = 3.0 * (lam_mix**2)
        ax.plot(t, C, label=r"$\mu={}$".format(mu))
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (two qubits)")
    ax.set_title(r"PMME: $\alpha=%.2f$, $\gamma_{\mathrm{M}}=%.1f$"%(alpha,gammaM))
    ax.legend()
    savefig(fig, "figures/fig5_5_PM_2q.pdf")

def fig5_6_PMME_1t(mu=0.6, alpha=0.15, gammaM=1.0, tmax=20.0):
    t = time_grid(tmax)
    A = np.abs(F_PMME(t, alpha, gammaM))
    lam_mix = (1.0 - mu)*(A**2) + mu*A
    C = 2.0 * lam_mix
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, C)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (single qutrit)")
    ax.set_title(r"PMME: $\mu=%.1f$, $\alpha=%.2f$, $\gamma_{\mathrm{M}}=%.1f$"%(mu,alpha,gammaM))
    savefig(fig, "figures/fig5_6_PM_1t.pdf")

# --- Critical surface (here identically zero for JC/PMME) ---
def fig5_8_mucrit_corrected(lam_JC=0.3, alpha_PM=0.15, gammaM=1.0, tmax=20.0):
    t = time_grid(tmax)
    mu_c_JC = np.zeros_like(t)
    mu_c_PM = np.zeros_like(t)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, mu_c_JC, label="JC")
    ax.plot(t, mu_c_PM, linestyle="--", label="PMME")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\mu_c(t)$")
    ax.set_ylim(-0.02, 0.12)
    ax.set_title(r"Critical surface $\mu_c(t)$ (JC/PMME: here $\mu_c\equiv 0$)")
    ax.legend()
    savefig(fig, "figures/fig5_8_mucrit_corrected.pdf")

# =========================
#     T-STAR HELPERS
# (separate; no plotting)
# =========================
def lambda_star(n1, n2):
    """λ* = ((2^{n1}-1)/(2^{n2}-1))^{1/(n2-n1)}"""
    if n1 == n2:
        raise ValueError("n1 and n2 must differ.")
    return ((2**n1 - 1.0)/(2**n2 - 1.0))**(1.0/(n2 - n1))

def t_star(mu, Gamma, n1, n2):
    """Correct closed-form t* (no plotting). Handles μ=1 separately."""
    lam_star = lambda_star(n1, n2)
    if mu == 1.0:
        return -(1.0/Gamma)*np.log(lam_star)
    y = (-mu + np.sqrt(mu**2 + 4.0*(1.0 - mu)*lam_star)) / (2.0*(1.0 - mu))
    return -(1.0/Gamma)*np.log(y)

def write_tstar_table(mu=0.6, Gamma=0.2, n_list=(1,2,4,6,8,10),
                      outfile="figures/tstar_table.csv"):
    """Compute t* for successive pairs in n_list and write a CSV table."""
    ensure_dir(os.path.dirname(outfile) or ".")
    pairs = [(n_list[i], n_list[i+1]) for i in range(len(n_list)-1)]
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("mu,Gamma,n1,n2,lambda_star,t_star\n")
        for (n1, n2) in pairs:
            lam_star = lambda_star(n1, n2)
            tstar = t_star(mu, Gamma, n1, n2)
            f.write(f"{mu},{Gamma},{n1},{n2},{lam_star:.12g},{tstar:.12g}\n")
    return outfile

# =========================
#         DRIVER
# =========================
def make_all():
    fig3_1a_pauli2q_compare(Gamma=0.2, mu=0.6, tmax=25.0)
    fig3_2_pauli_nq(Gamma=0.2, mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)
    fig3_3a_pauli1t_qutrit(Gamma=0.2, mu=0.6, tmax=25.0)
    fig3_4_pauli_nt_qutrit(Gamma=0.2, mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)

    fig4_1_GAD_2q(mu_list=(0.0,0.3,0.6,1.0), tmax=25.0)
    fig4_2_GAD_nq(mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)
    fig4_3_GAD_nt(mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)
    fig4_4_GAD_temp_single_qubit(gamma_const=0.3, num=501)

    fig5_1a_JC_2q(mu_list=(0.0,0.3,0.6,1.0), lam=0.3, gamma0=1.0, tmax=20.0)
    fig5_2_JC_nq(mu=0.6, lam=0.3, gamma0=1.0, n_list=(1,2,4,6,8,10), tmax=20.0)
    fig5_3_JC_1t(mu=0.6, lam=0.3, gamma0=1.0, tmax=20.0)

    fig5_5_PMME_2q(mu_list=(0.0,0.3,0.6,1.0), alpha=0.15, gammaM=1.0, tmax=20.0)
    fig5_6_PMME_1t(mu=0.6, alpha=0.15, gammaM=1.0, tmax=20.0)

    fig5_8_mucrit_corrected(lam_JC=0.3, alpha_PM=0.15, gammaM=1.0, tmax=20.0)

def parse_args():
    p = argparse.ArgumentParser(description="Generate all paper figures (PDF) into ./figures/")
    p.add_argument("--subset", type=str, default="all",
                   help="one of: all, pauli, gad, jc, pmme, mucrit, tstar")
    return p.parse_args()

def make_subset(which):
    which = which.lower()
    if which == "all":
        make_all()
    elif which == "pauli":
        fig3_1a_pauli2q_compare()
        fig3_2_pauli_nq()
        fig3_3a_pauli1t_qutrit()
        fig3_4_pauli_nt_qutrit()
    elif which == "gad":
        fig4_1_GAD_2q()
        fig4_2_GAD_nq()
        fig4_3_GAD_nt()
        fig4_4_GAD_temp_single_qubit()
    elif which == "jc":
        fig5_1a_JC_2q()
        fig5_2_JC_nq()
        fig5_3_JC_1t()
    elif which == "pmme":
        fig5_5_PMME_2q()
        fig5_6_PMME_1t()
    elif which == "mucrit":
        fig5_8_mucrit_corrected()
    elif which == "tstar":
        # Standalone: compute t* table (no plotting)
        out = write_tstar_table(mu=0.6, Gamma=0.2, n_list=(1,2,4,6,8,10),
                                outfile="figures/tstar_table.csv")
        print("Wrote:", out)
    else:
        raise ValueError("unknown subset: "+which)

if __name__ == "__main__":
    args = parse_args()
    make_subset(args.subset)
