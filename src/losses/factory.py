def build_loss_functions(
    args,
    hori_translation,
    verti_translation,
    label_mode,
    fusion_enabled,
    fusion_branches='io',
):
    from connect_loss import connect_loss, normalize_final_decision

    final_decision = normalize_final_decision(getattr(args, 'final_decision', 'max'))

    if fusion_enabled:
        has_middle = 'm' in fusion_branches
        has_outer  = 'o' in fusion_branches
        has_scale3 = 'p' in fusion_branches
        has_scale4 = 'f' in fusion_branches

        loss_func = connect_loss(
            args,
            hori_translation,
            verti_translation,
            label_mode=label_mode,
            conn_num=8,
            sigma=args.sigma,
            conn_layout="standard8",
            final_decision=final_decision,
        )
        # Non-standard auxiliary layouts keep Bilateral_voting_kxk's native max;
        # final_decision controls the standard8 fused/main path.
        loss_func_middle = connect_loss(
            args,
            hori_translation,
            verti_translation,
            label_mode=label_mode,
            conn_num=8,
            sigma=args.sigma,
            conn_layout="mid8",
            final_decision="max",
        ) if has_middle else None
        loss_func_outer = connect_loss(
            args,
            hori_translation,
            verti_translation,
            label_mode=label_mode,
            conn_num=8,
            sigma=args.sigma,
            conn_layout="out8",
            final_decision="max",
        ) if has_outer else None
        loss_func_scale3 = connect_loss(
            args,
            hori_translation,
            verti_translation,
            label_mode=label_mode,
            conn_num=8,
            sigma=args.sigma,
            conn_layout="scale3_8",
            final_decision="max",
        ) if has_scale3 else None
        loss_func_scale4 = connect_loss(
            args,
            hori_translation,
            verti_translation,
            label_mode=label_mode,
            conn_num=8,
            sigma=args.sigma,
            conn_layout="scale4_8",
            final_decision="max",
        ) if has_scale4 else None
        return (
            loss_func,
            loss_func_middle,
            loss_func_outer,
            loss_func_scale3,
            loss_func_scale4,
        )

    loss_func = connect_loss(
        args,
        hori_translation,
        verti_translation,
        label_mode=label_mode,
        conn_num=args.conn_num,
        sigma=args.sigma,
        conn_layout=getattr(args, "conn_layout", None),
        final_decision=final_decision,
    )
    return loss_func, None, None, None, None
