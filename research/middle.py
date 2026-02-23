@torch.no_grad()
def prune_linear_snr_2of4(layer, st: GramStat, score_mode: str, eps: float, refit: bool, ridge: float, horizon: int = 96, nf_hi: float = 0.25):
    W = layer.weight.data
    O, C = W.shape
    Ggroups = C // 4
    Cg = Ggroups * 4
    if Cg == 0:
        return
    device = W.device
    dtype = W.dtype

    Wg = W[:, :Cg].view(O, Ggroups, 4)
    Gs = (st.Gsig / max(st.Csig, 1e-6)).to(device=device, dtype=dtype)
    if st.Gspec is not None and st.Cspec > 0.0:
        Ga = (st.Gspec / max(st.Cspec, 1e-6)).to(device=device, dtype=dtype)
    else:
        Ga = (st.Gact / max(st.Cact, 1e-6)).to(device=device, dtype=dtype)

    use_noise = (score_mode == "sn_ratio2") and (st.Gnoi is not None) and (st.Cnoi > 0.0)
    if use_noise:
        Gn = (st.Gnoi / max(st.Cnoi, 1e-6)).to(device=device, dtype=dtype)
    else:
        Gn = None
        if score_mode == "sn_ratio2":
            score_mode = "ratio"

    # True Competitive MoE: evaluate ALL experts against held-out validation data
    if score_mode == "unified":
        import math
        # ── Pre-compute shared quantities ────────────────────────────────────
        damp_s = 0.01 * torch.mean(torch.diagonal(Gs, dim1=1, dim2=2))
        H_s    = Gs + damp_s * torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        Hinv_s = torch.inverse(H_s)
        diag_s = torch.diagonal(Hinv_s, dim1=1, dim2=2)               # [G, 4]

        damp_a = 0.01 * torch.mean(torch.diagonal(Ga, dim1=1, dim2=2))
        H_a    = Ga + damp_a * torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        Hinv_a = torch.inverse(H_a)
        diag_a = torch.diagonal(Hinv_a, dim1=1, dim2=2)               # [G, 4]

        act_diag = torch.diagonal(Ga, dim1=1, dim2=2).clamp_min(1e-8)  # [G, 4]

        log_ridge = math.log10(max(ridge, 1e-10))
        t_ridge   = max(0.0, min(1.0, (log_ridge - (-3.0)) / 1.0))
        w_obs, w_ratio, w_mag = 0.25+0.6*t_ridge, 0.5-0.4*t_ridge, 0.25-0.2*t_ridge

        def znorm(t): return (t - t.mean()) / (t.std() + 1e-9)

        masks_t = PAIR_MASKS.to(device=device, dtype=dtype)
        # E1: MAG
        scores_mag = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        # E2: Wanda
        wanda_imp = (Wg ** 2) * act_diag.unsqueeze(0)
        scores_wanda = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        # E4: OBS (Gact)
        obs_imp_a = (Wg ** 2) / (diag_a.unsqueeze(0) + 1e-10)
        scores_obs = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        # E3: SNR
        snr_ratio = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        snr_mag   = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)
        snr_obsig = torch.zeros((O, Ggroups, 6), device=device, dtype=dtype)

        for k in range(6):
            mk = masks_t[k].view(1, 1, 4)
            Wk = Wg * mk
            Wd = Wg * (1.0 - mk)
            scores_mag[:, :, k]   = Wk.abs().sum(dim=2)
            scores_wanda[:, :, k] = (wanda_imp * mk).sum(dim=2)
            scores_obs[:, :, k]   = (obs_imp_a * mk).sum(dim=2)
            
            Tk_s, Td_s = torch.einsum("ogc,gcd->ogd", Wk, Gs), torch.einsum("ogc,gcd->ogd", Wd, Gs)
            snr_ratio[:, :, k] = (Tk_s * Wk).sum(dim=2) / ((Td_s * Wd).sum(dim=2) + eps)
            snr_mag[:, :, k]   = Wk.abs().sum(dim=2)
            snr_obsig[:, :, k] = (((Wk ** 2) / (diag_s.unsqueeze(0) + 1e-10)) * mk).sum(dim=2)

        scores_snr = (w_ratio * znorm(snr_ratio.reshape(-1, 6)).reshape(O, Ggroups, 6)
                    + w_mag   * znorm(snr_mag.reshape(-1, 6)).reshape(O, Ggroups, 6)
                    + w_obs   * znorm(snr_obsig.reshape(-1, 6)).reshape(O, Ggroups, 6))

        expert_bestks = [
            torch.argmax(scores_mag, dim=2), torch.argmax(scores_wanda, dim=2),
            torch.argmax(scores_snr, dim=2), torch.argmax(scores_obs, dim=2)
        ]
        expert_names = ["MAG", "Wanda", "SNR", "OBS"]

        # ── Empirical Validation ─────────────────────────────────────────────
        if st.X_val is not None:
            X_val_d = st.X_val.to(device=device, dtype=dtype)
            Y_val_dense = (X_val_d @ W[:, :Cg].T).detach()
        else:
            X_val_d = torch.randn((1, Cg), device=device, dtype=dtype)
            Y_val_dense = (X_val_d @ W[:, :Cg].T).detach()

        G_refit, ridge_refit = Gs, ridge
        H_reg = G_refit + ridge_refit * torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        B = torch.einsum("ogc,gcd->ogd", Wg, H_reg)
        invs = []
        eye2 = torch.eye(2, device=device, dtype=dtype).view(1, 2, 2)
        for k in range(6):
            i, j = PAIRS[k].tolist()
            G_sub = G_refit[:, [i, j]][:, :, [i, j]]
            invs.append(torch.inverse(G_sub + ridge_refit * eye2))
        invs = torch.stack(invs, dim=0)
        g_idx = torch.arange(Ggroups, device=device).view(1, Ggroups).expand(O, Ggroups)

        import sys
        REFIT_PENALTY = 1.25  # Requires 25% improvement
        SNR_BIAS      = 0.95  # Robustness bias toward SNR Mask-only
        
        mask_mses = []
        for idx_e, bbestk in enumerate(expert_bestks):
            pair_idx = PAIRS.to(device)[bbestk]
            mask = torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool).scatter_(2, pair_idx, True)
            W_m = torch.where(mask, Wg, torch.zeros_like(Wg))
            mse_m = torch.mean(((X_val_d @ W_m.view(O, Cg).T) - Y_val_dense)**2).item()
            mask_mses.append(mse_m)

        best_mask_mse = min(mask_mses)
        best_mask_idx = mask_mses.index(best_mask_mse)
        
        best_mse = best_mask_mse
        final_expert = expert_names[best_mask_idx]
        final_refit = False

        if refit:
            for idx_e, bbestk in enumerate(expert_bestks):
                W_r = torch.zeros_like(Wg)
                for k in range(6):
                    sel = (bbestk == k)
                    if not torch.any(sel): continue
                    i,j = PAIRS[k].tolist()
                    bsel = torch.stack([B[:,:,i][sel], B[:,:,j][sel]], dim=1)
                    invsel = invs[k, g_idx[sel], :, :]
                    u = torch.bmm(invsel, bsel.unsqueeze(2)).squeeze(2)
                    W_r[:,:,i][sel], W_r[:,:,j][sel] = u[:, 0], u[:, 1]
                
                mse_r = torch.mean(((X_val_d @ W_r.view(O, Cg).T) - Y_val_dense)**2).item()
                if (mse_r * REFIT_PENALTY) < best_mask_mse:
                    if mse_r < best_mse:
                        best_mse, final_expert, final_refit = mse_r, expert_names[idx_e], True

        if best_mse > (mask_mses[2] * SNR_BIAS):
            best_mse, final_expert, final_refit = mask_mses[2], "SNR", False

        winner_bestk = expert_bestks[expert_names.index(final_expert)]
        if not final_refit:
            p_idx = PAIRS.to(device)[winner_bestk]
            Wnew = torch.where(torch.zeros((O,Ggroups,4), device=device, dtype=torch.bool).scatter_(2, p_idx, True), Wg, torch.zeros_like(Wg))
        else:
            Wnew = torch.zeros_like(Wg)
            for k in range(6):
                sel = (winner_bestk == k)
                if not torch.any(sel): continue
                i,j = PAIRS[k].tolist()
                bsel = torch.stack([B[:,:,i][sel], B[:,:,j][sel]], dim=1)
                invsel = invs[k, g_idx[sel], :, :]
                u = torch.bmm(invsel, bsel.unsqueeze(2)).squeeze(2)
                Wnew[:,:,i][sel], Wnew[:,:,j][sel] = u[:, 0], u[:, 1]

        sys.stderr.write(f"[moe] Winner: {final_expert} Refit={final_refit} MSE_val={best_mse:.6f}\n")

    else:
        expert_bestks_l = [torch.argmax(torch.empty((O, Ggroups, 6), device=device, dtype=dtype), dim=2)] # dummy
        masks = PAIR_MASKS.to(device=device, dtype=dtype)
        scores = torch.empty((O, Ggroups, 6), device=device, dtype=dtype)
        def znorm(t): return (t - t.mean()) / (t.std() + 1e-9)
        damp_s = 0.01 * torch.mean(torch.diagonal(Gs, dim1=1, dim2=2))
        H_s = Gs + damp_s * torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        Hinv_s = torch.inverse(H_s)
        diag_s = torch.diagonal(Hinv_s, dim1=1, dim2=2)
        obs_imp_s = (Wg ** 2) / (diag_s.unsqueeze(0) + 1e-10)

        for k in range(6):
            mk = masks[k].view(1, 1, 4)
            Wk, Wd = Wg * mk, Wg * (1.0 - mk)
            Tk_s, Td_s = torch.einsum("ogc,gcd->ogd", Wk, Gs), torch.einsum("ogc,gcd->ogd", Wd, Gs)
            if score_mode == "keep": scores[:,:,k] = (Tk_s * Wk).sum(dim=2)
            elif score_mode == "ratio": scores[:,:,k] = (Tk_s * Wk).sum(dim=2) / ((Td_s * Wd).sum(dim=2) + eps)
            else:
                Tk_n, Td_n = torch.einsum("ogc,gcd->ogd", Wk, Gn), torch.einsum("ogc,gcd->ogd", Wd, Gn)
                scores[:,:,k] = ((Tk_s*Wk).sum(dim=2)/((Td_s*Wd).sum(dim=2)+eps)) / ((Tk_n*Wk).sum(dim=2)/((Td_n*Wd).sum(dim=2)+eps)+eps)
        
        bestk = torch.argmax(scores, dim=2)
        if not refit:
            pair_idx = PAIRS.to(device)[bestk]
            mask = torch.zeros((O, Ggroups, 4), device=device, dtype=torch.bool).scatter_(2, pair_idx, True)
            Wnew = torch.where(mask, Wg, torch.zeros_like(Wg))
        else:
            H_reg = Gs + ridge * torch.eye(4, device=device, dtype=dtype).unsqueeze(0); B_l = torch.einsum("ogc,gcd->ogd", Wg, H_reg)
            invs_l = []
            eye2 = torch.eye(2, device=device, dtype=dtype).view(1, 2, 2)
            for k in range(6):
                i, j = PAIRS[k].tolist(); G_sub = Gs[:, [i, j]][:, :, [i, j]]; invs_l.append(torch.inverse(G_sub + ridge * eye2))
            invs_l = torch.stack(invs_l, dim=0); g_idx = torch.arange(Ggroups, device=device).view(1, Ggroups).expand(O, Ggroups)
            Wnew = torch.zeros_like(Wg)
            for k in range(6):
                sel = (bestk == k)
                if not torch.any(sel): continue
                bsel = torch.stack([B_l[:,:,PAIRS[k,0]][sel], B_l[:,:,PAIRS[k,1]][sel]], dim=1)
                invsel = invs_l[k, g_idx[sel], :, :]; u = torch.bmm(invsel, bsel.unsqueeze(2)).squeeze(2)
                Wnew[:,:,PAIRS[k,0]][sel], Wnew[:,:,PAIRS[k,1]][sel] = u[:, 0], u[:, 1]
    
    W[:, :Cg] = Wnew.view(O, Cg)
