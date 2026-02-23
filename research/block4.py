        Y_pool = Y_train[:args.calib_windows]

    X_test_all, Y_test_all = make_windows(series, args.train_end, n_total, args.context, args.horizon, args.stride_test)
    ts = args.test_start_window
    if args.test_windows < 0:
        te = X_test_all.shape[0]
    else:
        te = min(ts + args.test_windows, X_test_all.shape[0])
    X_test, Y_test = X_test_all[ts:te], Y_test_all[ts:te]

    # Logs for fairness/transparency
    print(f"[split] total_rows={n_total} train_end={args.train_end} context={args.context} horizon={args.horizon}")
    print(f"[train] windows={X_train.shape[0]} calib_pool={X_pool.shape[0]}")
    req_tw = args.test_windows
    print(f"[test]  available={X_test_all.shape[0]} requested={req_tw} start={ts} using={X_test.shape[0]} stride_test={args.stride_test}")
    gram_budget = args.max_calls_per_layer * args.calib_batch
    eff_K = min(X_pool.shape[0], gram_budget)
    print(f"[calib] eval_batch={args.batch} calib_batch={args.calib_batch} gram_budget=max_calls_per_layer*calib_batch={args.max_calls_per_layer}*{args.calib_batch}={gram_budget} => effective_K={eff_K}")

    # Model
    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    torch_mod = find_torch_module(tfm)
    targets = select_linears(torch_mod, args.include_quantile_head, args.include_regex, args.exclude_regex)
    tfm.compile(timesfm.ForecastConfig(max_context=args.context, max_horizon=max(args.horizon, 256)))

    # Baseline eval
    if args.measure_time:
        pred_base, tsec = timed_forecast(tfm, X_test, args.horizon, args.batch)
        mse_b, mae_b = mse_mae(pred_base, Y_test)
        print(f"[baseline] MSE={mse_b:.6f} MAE={mae_b:.6f} | avg_batch_sec={tsec:.4f}")
    else:
        preds = []
        for i in range(0, len(X_test), args.batch):
            preds.append(forecast_timesfm_point(tfm, X_test[i:i+args.batch], args.horizon))
        pred_base = np.concatenate(preds, axis=0)
        mse_b, mae_b = mse_mae(pred_base, Y_test)
        print(f"[baseline] MSE={mse_b:.6f} MAE={mae_b:.6f}")


    # Decide if we need error computation
    need_errors = (args.calib_select == "topk") or (args.error_power != 0.0) or (args.score_mode == "sn_ratio2")

    if need_errors:
        print(f"[calib] computing errors/weights on pool: power={args.error_power} (batched={args.calib_batch}) ...")
        errors, err_ratio, weights = compute_errors_and_weights(
            tfm_model=tfm, X_pool=X_pool, Y_pool=Y_pool,
            horizon=args.horizon, calib_batch=args.calib_batch, error_power=args.error_power
        )
        print(f"[calib] weight_stats: min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f} (power={args.error_power})")
    else:
        errors = None
        err_ratio = np.ones((X_pool.shape[0],), dtype=np.float32)
        weights = np.ones((X_pool.shape[0],), dtype=np.float32)
        print(f"[calib] uniform weights (no labels used): power=0, select={args.calib_select}")

    # Select K windows
    rng = np.random.default_rng(args.seed)
    K = eff_K

    if args.calib_select in ("first", "last"):
        sel_idx = np.arange(K, dtype=np.int64)
    elif args.calib_select == "random":
        sel_idx = rng.choice(X_pool.shape[0], size=K, replace=False).astype(np.int64)
    elif args.calib_select == "topk":
        # top-K by error (equivalent to top-K by weight when power>0)
        if errors is None:
            raise RuntimeError("topk selection requires errors, but errors were not computed.")
        sel_idx = np.argsort(errors)[-K:].astype(np.int64)
    else:
        raise ValueError(f"Unknown calib_select: {args.calib_select}")

    X_sel = X_pool[sel_idx]

    # --- weights used for gram collection ---
    # Legacy behavior (single gram): w_sel = (err_ratio^error_power) over selected windows.
    # New behavior for score_mode=sn_ratio2: build *two* grams:
    #   signal weights  ~ err_ratio^(-sn_gamma)  (emphasize easy/predictable windows)
    #   noise weights   ~ err_ratio^(+sn_gamma)  (emphasize hard/high-error windows)
    w_sel = weights[sel_idx].astype(np.float32)
    er_sel = err_ratio[sel_idx].astype(np.float32)

    if args.score_mode == "sn_ratio2":
        ws = (er_sel ** (-float(args.sn_gamma))).astype(np.float32)
        wn = (er_sel ** ( float(args.sn_gamma))).astype(np.float32)
        # normalize to mean 1 for numerical stability / comparable counts
        ws = (ws / (ws.mean() + 1e-8)).astype(np.float32)
        wn = (wn / (wn.mean() + 1e-8)).astype(np.float32)
        w_sig_sel, w_noi_sel = ws, wn
        print(f"[calib] sn_ratio2 grams: sn_gamma={args.sn_gamma:g} | "
              f"sig_w(min/mean/max)=({ws.min():.3g}/{ws.mean():.3g}/{ws.max():.3g}) "
              f"noi_w(min/mean/max)=({wn.min():.3g}/{wn.mean():.3g}/{wn.max():.3g})")
    else:
        w_sig_sel, w_noi_sel = w_sel, None

    print(f"[calib] collecting grams: select={args.calib_select} pool={X_pool.shape[0]} K={K} (max_calls_per_layer={args.max_calls_per_layer})")

    gram_stats = collect_group_grams_signal_noise(
        tfm_model=tfm,
        targets=targets,
        X_sel=X_sel,
        w_sig_sel=w_sig_sel,
        w_noi_sel=w_noi_sel,
        horizon=args.horizon,
        calib_batch=args.calib_batch,
        sample_rows_per_call=args.sample_rows_per_call,
        max_calls_per_layer=args.max_calls_per_layer,
    )
    print(f"[calib] collected grams for {len(gram_stats)}/{len(targets)} layers")

    # Prune
    print(f"[prune] SNR 2:4: score_mode={args.score_mode}, refit={bool(args.refit)} ridge={args.ridge:g}")
    
    # Auto-adaptive ridge for unified mode based on global noise fraction
    effective_ridge = args.ridge
    if args.score_mode == "unified" and bool(args.refit):
        noise_fracs = []
        for name, layer in targets:
            st = gram_stats.get(name, None)
            if st is None or st.Csig <= 0.0 or st.sfm_sum is None or st.count == 0:
                continue
            cnt = float(max(st.count, 1))
            E_t = float((st.trend_energy / cnt).sum().item())
            E_s = float((st.season_energy / cnt).sum().item())
            E_n = float((st.noise_energy / cnt).sum().item())
            nf = E_n / (E_t + E_s + E_n + 1e-12)
            noise_fracs.append(nf)
        if noise_fracs:
            noise_fracs.sort()
            max_nf = noise_fracs[-1]
            avg_nf = sum(noise_fracs) / len(noise_fracs)
            # Map max noise fraction to ridge
            # We want clear separation: ETTm (~0.25) vs ETTh (>0.26) is tricky.
            # But earlier ETTm max_nf was 0.2631? And ETTh 0.2661?
            # Wait, the separation on pure max_nf is thin.
            # But the *impact* of that noise is what matters.
            # Let's stick to the mapping: nf=0.15 -> ridge, nf=0.30 -> ridge*1000
            if max_nf > 0.15:
                import math
                t = min(1.0, (max_nf - 0.15) / 0.15)
                effective_ridge = args.ridge * (10 ** (4.0 * t))  # 1e-5 → 1e-1 at t=1 (more aggressive)
            print(f"[unified-v2] noise_frac: avg={avg_nf:.4f} max={max_nf:.4f} → effective_ridge={effective_ridge:.2e} (base={args.ridge:.2e})")

    # Spectral diagnostics
    if args.score_mode in ("spectral", "unified"):
        c_layers = []
        for name, layer in targets:
            st = gram_stats.get(name, None)
            if st is None or st.Csig <= 0.0 or st.sfm_sum is None or st.count == 0:
                continue
            cnt = float(max(st.count, 1))
            sfm_avg = st.sfm_sum / cnt
            E_t = (st.trend_energy / cnt)
            E_s = (st.season_energy / cnt)
            E_n = (st.noise_energy / cnt)
            total_E = E_t.sum() + E_s.sum() + E_n.sum() + 1e-12
            conf = 1.0 - torch.clamp((sfm_avg - 0.2) / 0.4, 0.0, 1.0)
            c_val = float(conf.mean().item())
            c_layers.append(c_val)
        if c_layers:
            import statistics
            c_arr = c_layers
            print(f"[diag] C_layer stats over {len(c_arr)} layers: "
                  f"min={min(c_arr):.4f} max={max(c_arr):.4f} "
                  f"mean={statistics.mean(c_arr):.4f} median={statistics.median(c_arr):.4f} "
                  f"stdev={statistics.stdev(c_arr):.4f}" if len(c_arr) > 1 else f"[diag] C_layer: {c_arr[0]:.4f}")
            # Print energy band summary from first layer
            st0 = gram_stats.get(targets[0][0], None)
            if st0 is not None and st0.sfm_sum is not None:
                cnt = float(max(st0.count, 1))
                E_t = float((st0.trend_energy / cnt).sum().item())
                E_s = float((st0.season_energy / cnt).sum().item())
                E_n = float((st0.noise_energy / cnt).sum().item())
                total = E_t + E_s + E_n + 1e-12
                print(f"[diag] Energy bands (layer0): trend={E_t/total:.1%} season={E_s/total:.1%} noise={E_n/total:.1%}")

    for name, layer in targets:
        st = gram_stats.get(name, None)
        if st is None or st.Csig <= 0.0:
            continue
        prune_linear_snr_2of4(layer, st, args.score_mode, args.eps, bool(args.refit), effective_ridge, horizon=args.horizon, nf_hi=args.nf_hi)


    # Eval pruned
    if args.measure_time:
        pred_p, tsec2 = timed_forecast(tfm, X_test, args.horizon, args.batch)
        mse_p, mae_p = mse_mae(pred_p, Y_test)
        print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f} | avg_batch_sec={tsec2:.4f}")
    else:
        preds = []
        for i in range(0, len(X_test), args.batch):
            preds.append(forecast_timesfm_point(tfm, X_test[i:i+args.batch], args.horizon))
        pred_p = np.concatenate(preds, axis=0)
        mse_p, mae_p = mse_mae(pred_p, Y_test)
        print(f"[snr-2of4-refit] MSE={mse_p:.6f} MAE={mae_p:.6f}")

    print(f"[delta] ΔMSE={(mse_p - mse_b):+.6f}  ΔMAE={(mae_p - mae_b):+.6f}")

if __name__ == "__main__":
    main()
