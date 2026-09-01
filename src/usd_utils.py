GRIPPER_PRIM_PATH = "/World/ur10e/ee_link/Robotiq_2F_85"


def expand_gripper_visual_instances(stage) -> int:
    """Expand instanced gripper meshes to avoid stale Fabric prototype references."""
    from pxr import Usd

    gripper_prim = stage.GetPrimAtPath(GRIPPER_PRIM_PATH)
    if not gripper_prim.IsValid():
        raise RuntimeError(f"Gripper prim not found: {GRIPPER_PRIM_PATH}")

    visual_instances = [
        prim for prim in Usd.PrimRange(gripper_prim) if prim.GetName() == "visuals" and prim.IsInstanceable()
    ]
    for prim in visual_instances:
        prim.SetInstanceable(False)
    return len(visual_instances)
