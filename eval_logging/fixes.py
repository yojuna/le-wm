"""Workarounds for stable-worldmodel rollout quirks in live eval."""


def install_mpc_buffer_fix(world) -> None:
    """Clear stale ``_needs_flush`` after planning.

    WorldModelPolicy pops ``_needs_flush`` from a copied info dict, not from
    ``world.infos``. After the first auto-reset between episodes the flag
    therefore sticks and the MPC action buffer is cleared on every step,
    causing one CEM solve per env step (~0.9s each with default settings).
    """
    orig_get_actions = world._get_actions

    def patched_get_actions():
        actions = orig_get_actions()
        if isinstance(world.infos, dict):
            world.infos.pop("_needs_flush", None)
        return actions

    world._get_actions = patched_get_actions
