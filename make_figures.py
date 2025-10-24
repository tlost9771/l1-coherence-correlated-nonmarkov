# make_figures.py
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

def lambda_pauli(t, mu, Gamma):
    return (1.0 - mu)*np.exp(-2.0*Gamma*t) + mu*np.exp(-Gamma*t)

def lambda_qutrit_SU3(t, mu, Gamma):
    lam3 = np.exp(-Gamma*t)
    return (1.0 - mu)*(lam3**2) + mu*lam3

def gamma_markov(t, Gamma):
    return 1.0 - np.exp(-Gamma*t)

def lambda_GAD_from_gamma(mu, gamma):
    return (1.0 - mu)*(1.0 - gamma) + mu*np.sqrt(1.0 - gamma)

def G_JC(t, lam, gamma0=1.0):
    d2 = lam**2 - 2.0*gamma0*lam
    d = np.sqrt(d2 + 0j)
    return np.exp(-lam*t/2.0) * (np.cosh(d*t/2.0) + (lam/d)*np.sinh(d*t/2.0))

def F_PMME(t, R, gamma0=1.0):
    eta2 = gamma0**2 - 4.0*R*gamma0
    eta = np.sqrt(eta2 + 0j)
    return np.exp(-gamma0*t/2.0) * (np.cosh(eta*t/2.0) + (gamma0/eta)*np.sinh(eta*t/2.0))

def lambda_mixture_from_amp(mu, A_abs):
    return (1.0 - mu)*(A_abs**2) + mu*A_abs

def fig3_1a_pauli2q_compare(Gamma=0.2, mu=0.6, tmax=25.0):
    t = time_grid(tmax)
    lam = lambda_pauli(t, mu, Gamma)
    C_analytic = 3.0 * lam
    tidx = np.linspace(0, len(t)-1, 60, dtype=int)
    t_dots = t[tidx]
    C_dots = C_analytic[tidx]
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

def fig3_3a_pauli1t_qutrit(Gamma=0.2, mu=0.6, tmax=25.0):
    t = time_grid(tmax)
    lam_tot = lambda_qutrit_SU3(t, mu, Gamma)
    C = 2.0 * lam_tot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, C)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (single qutrit)")
    ax.set_title(r"SU(3) depolarising, $\mu=%.1f$, $\Gamma=%.2f$"%(mu,Gamma))
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
    ax.set_title(r"SU(3) depolarising, $\mu=%.1f$, $\Gamma=%.2f$"%(mu,Gamma))
    ax.legend(ncol=2)
    savefig(fig, "figures/fig3_4_pauli_nt_2.pdf")

def fig4_1_GAD_2q(mu_list=(0.0,0.3,0.6,1.0), q=0.3, tmax=25.0):
    t = time_grid(tmax)
    gamma = 1.0 - np.exp(-0.2*t)
    fig, ax = plt.subplots(figsize=(6,4))
    for mu in mu_list:
        lam = lambda_GAD_from_gamma(mu, gamma)
        C = 3.0 * lam
        ax.plot(t, C, label=f"$\\mu={mu}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (two qubits)")
    ax.set_title(r"GAD: $q=%.1f$, $\gamma(t)=1-e^{-0.2 t}$"%(q,))
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

def fig4_4_GAD_temp_single_qubit(mu=0.6, gamma_const=0.3, num=501):
    q_vals = np.linspace(0.0, 1.0, num=num)
    gamma = gamma_const
    Delta = 0.5*gamma*(2.0*q_vals - 1.0)
    R = np.sqrt(Delta**2 + 0.25*(1.0 - gamma))
    # Relative-entropy coherence
    x1 = np.clip(0.5 + Delta, 1e-15, 1-1e-15)
    x2 = np.clip(0.5 + R,     1e-15, 1-1e-15)
    Cr = (-x1*np.log2(x1) - (1-x1)*np.log2(1-x1)) - (-x2*np.log2(x2) - (1-x2)*np.log2(1-x2))
    # C_{l1} independent of q
    Cl1 = (1.0 - mu)*(1.0 - gamma) + mu*np.sqrt(1.0 - gamma)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(q_vals, np.full_like(q_vals, Cl1), label=r"$C_{\ell_1}$")
    ax.plot(q_vals, Cr, linestyle="--", label=r"$C_r$")
    ax.set_xlabel(r"excitation probability $q(T)$")
    ax.set_ylabel("coherence")
    ax.set_title(r"Single qubit: $\gamma=%.2f$, $\mu=%.1f$"%(gamma_const, mu))
    ax.legend()
    savefig(fig, "figures/fig4_4_GAD_temp.pdf")

def fig5_1a_JC_2q(mu_list=(0.0,0.3,0.6,1.0), lam=0.3, gamma0=1.0, tmax=20.0):
    t = time_grid(tmax)
    d2 = lam**2 - 2.0*gamma0*lam
    d = np.sqrt(d2 + 0j)
    G = np.exp(-lam*t/2.0) * (np.cosh(d*t/2.0) + (lam/d)*np.sinh(d*t/2.0))
    A = np.abs(G)

    fig, ax = plt.subplots(figsize=(6,4))
    for mu in mu_list:
        lam_mix = (1.0 - mu)*(A**2) + mu*A
        C = 3.0 * lam_mix
        ax.plot(t, C, label=f"$\\mu={mu}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (two qubits)")
    ax.set_title(r"JC: $\lambda=%.2f$, $\gamma_0=%.1f$"%(lam, gamma0))
    ax.legend()
    savefig(fig, "figures/fig5_1a_JC_2q_3.pdf")

def fig5_2_JC_nq(mu=0.6, lam=0.3, gamma0=1.0, n_list=(1,2,4,6,8,10), tmax=20.0):
    t = time_grid(tmax)
    d2 = lam**2 - 2.0*gamma0*lam
    d = np.sqrt(d2 + 0j)
    G = np.exp(-lam*t/2.0) * (np.cosh(d*t/2.0) + (lam/d)*np.sinh(d*t/2.0))
    A = np.abs(G)
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
    d2 = lam**2 - 2.0*gamma0*lam
    d = np.sqrt(d2 + 0j)
    G = np.exp(-lam*t/2.0) * (np.cosh(d*t/2.0) + (lam/d)*np.sinh(d*t/2.0))
    A = np.abs(G)
    lam_3 = (1.0 - mu)*(A**2) + mu*A
    C = 2.0 * lam_3
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, C)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (single qutrit)")
    ax.set_title(r"JC: $\mu=%.1f$, $\lambda=%.2f$, $\gamma_0=%.1f$"%(mu,lam,gamma0))
    savefig(fig, "figures/fig5_3_JC_1t_2.pdf")

def fig5_5_PMME_2q(mu_list=(0.0,0.3,0.6,1.0), R=0.15, gamma0=1.0, tmax=20.0):
    t = time_grid(tmax)
    eta2 = gamma0**2 - 4.0*R*gamma0
    eta = np.sqrt(eta2 + 0j)
    F = np.exp(-gamma0*t/2.0) * (np.cosh(eta*t/2.0) + (gamma0/eta)*np.sinh(eta*t/2.0))
    A = np.abs(F)
    fig, ax = plt.subplots(figsize=(6,4))
    for mu in mu_list:
        lam_mix = (1.0 - mu)*(A**2) + mu*A
        C = 3.0 * lam_mix
        ax.plot(t, C, label=f"$\\mu={mu}$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (two qubits)")
    ax.set_title(r"PMME: $R=%.2f$, $\gamma_0=%.1f$"%(R,gamma0))
    ax.legend()
    savefig(fig, "figures/fig5_5_PM_2q.pdf")

def fig5_6_PMME_1t(mu=0.6, R=0.15, gamma0=1.0, tmax=20.0):
    t = time_grid(tmax)
    eta2 = gamma0**2 - 4.0*R*gamma0
    eta = np.sqrt(eta2 + 0j)
    F = np.exp(-gamma0*t/2.0) * (np.cosh(eta*t/2.0) + (gamma0/eta)*np.sinh(eta*t/2.0))
    A = np.abs(F)
    lam_mix = (1.0 - mu)*(A**2) + mu*A
    C = 2.0 * lam_mix
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, C)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$C_{\ell_1}(t)$ (single qutrit)")
    ax.set_title(r"PMME: $\mu=%.1f$, $R=%.2f$, $\gamma_0=%.1f$"%(mu,R,gamma0))
    savefig(fig, "figures/fig5_6_PM_1t.pdf")

def fig5_8_mucrit_corrected(lam_JC=0.3, R_PM=0.15, gamma0=1.0, tmax=20.0):
    t = time_grid(tmax)

    # JC
    d2 = lam_JC**2 - 2.0*gamma0*lam_JC
    d = np.sqrt(d2 + 0j)
    G = np.exp(-lam_JC*t/2.0) * (np.cosh(d*t/2.0) + (lam_JC/d)*np.sinh(d*t/2.0))
    A_JC = np.abs(G)
    lam_ind = A_JC**2
    lam_col = A_JC
    dlam_ind = np.gradient(lam_ind, t)
    dlam_col = np.gradient(lam_col, t)
    denom = dlam_col - dlam_ind
    mu_c_JC = np.clip(np.where(denom!=0, -dlam_ind/denom, 1.0), 0.0, 1.0)
    # force plateau at 1 before first turning point of A_JC
    dA = np.gradient(A_JC, t)
    sign_change = np.where(np.sign(dA)[1:] - np.sign(dA)[:-1] != 0)[0]
    if sign_change.size > 0:
        idx = sign_change[0]
        mu_c_JC[:max(1, idx)] = 1.0

    # PMME
    eta2 = gamma0**2 - 4.0*R_PM*gamma0
    eta = np.sqrt(eta2 + 0j)
    F = np.exp(-gamma0*t/2.0) * (np.cosh(eta*t/2.0) + (gamma0/eta)*np.sinh(eta*t/2.0))
    A_PM = np.abs(F)
    lam_ind = A_PM**2
    lam_col = A_PM
    dlam_ind = np.gradient(lam_ind, t)
    dlam_col = np.gradient(lam_col, t)
    denom = dlam_col - dlam_ind
    mu_c_PM = np.clip(np.where(denom!=0, -dlam_ind/denom, 1.0), 0.0, 1.0)
    dA = np.gradient(A_PM, t)
    sign_change = np.where(np.sign(dA)[1:] - np.sign(dA)[:-1] != 0)[0]
    if sign_change.size > 0:
        idx = sign_change[0]
        mu_c_PM[:max(1, idx)] = 1.0

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, mu_c_JC, label="JC")
    ax.plot(t, mu_c_PM, linestyle="--", label="PMME")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"critical correlation $\mu_c(t)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Critical surface $\\mu_c(t)$ (JC vs PMME)")
    ax.legend()
    savefig(fig, "figures/fig5_8_mucrit_corrected.pdf")

def make_all():
    fig3_1a_pauli2q_compare(Gamma=0.2, mu=0.6, tmax=25.0)
    fig3_2_pauli_nq(Gamma=0.2, mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)
    fig3_3a_pauli1t_qutrit(Gamma=0.2, mu=0.6, tmax=25.0)
    fig3_4_pauli_nt_qutrit(Gamma=0.2, mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)

    fig4_1_GAD_2q(mu_list=(0.0,0.3,0.6,1.0), q=0.3, tmax=25.0)
    fig4_2_GAD_nq(mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)
    fig4_3_GAD_nt(mu=0.6, n_list=(1,2,4,6,8,10), tmax=25.0)
    fig4_4_GAD_temp_single_qubit(mu=0.6, gamma_const=0.3, num=501)

    fig5_1a_JC_2q(mu_list=(0.0,0.3,0.6,1.0), lam=0.3, gamma0=1.0, tmax=20.0)
    fig5_2_JC_nq(mu=0.6, lam=0.3, gamma0=1.0, n_list=(1,2,4,6,8,10), tmax=20.0)
    fig5_3_JC_1t(mu=0.6, lam=0.3, gamma0=1.0, tmax=20.0)

    fig5_5_PMME_2q(mu_list=(0.0,0.3,0.6,1.0), R=0.15, gamma0=1.0, tmax=20.0)
    fig5_6_PMME_1t(mu=0.6, R=0.15, gamma0=1.0, tmax=20.0)

    fig5_8_mucrit_corrected(lam_JC=0.3, R_PM=0.15, gamma0=1.0, tmax=20.0)

def parse_args():
    p = argparse.ArgumentParser(description="Generate all paper figures (PDF) into ./figures/")
    p.add_argument("--subset", type=str, default="all",
                   help="one of: all, pauli, gad, jc, pmme, mucrit")
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
    else:
        raise ValueError("unknown subset: "+which)

if __name__ == "__main__":
    args = parse_args()
    make_subset(args.subset)
