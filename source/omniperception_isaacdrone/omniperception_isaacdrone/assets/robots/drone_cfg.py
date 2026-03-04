from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

_THIS_DIR = Path(__file__).resolve().parent
_USD_PATH = str(_THIS_DIR / "cf2x.usd")

# =============================================================================
# IMPORTANT: Mass convention for articulations (multi-body)
# -----------------------------------------------------------------------------
# IsaacLab applies UsdFileCfg.mass_props using `modify_mass_properties` decorated
# with `apply_nested`, which sets the mass on *every* child prim (link) that has
# MassAPI under the spawned prim path.
#
# Your cf2x.usd has multiple bodies (e.g., body + 4 props). If you set mass=0.25
# directly, each body becomes 0.25 kg -> total mass becomes ~1.25 kg.
# Then your controller (assuming 0.25 kg total) will output ~5x too little thrust.
#
# We keep DRONE_MASS as the desired TOTAL mass, and apply DRONE_LINK_MASS per link.
# =============================================================================

DRONE_MASS: float = 0.25  # kg (desired TOTAL mass of the whole drone)

# NOTE: Based on your runtime print:
# body names: ['body', 'm1_prop', 'm2_prop', 'm3_prop', 'm4_prop']  -> 5 bodies
CF2X_NUM_BODIES: int = 5

# Per-link mass written to USD (so total ≈ DRONE_MASS)
DRONE_LINK_MASS: float = DRONE_MASS / CF2X_NUM_BODIES


@configclass
class Cf2xDroneCfg(ArticulationCfg):
    """Crazyflie 2.x like drone config"""

    spawn = sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,

        # Apply per-link mass (NOT total mass) to avoid multiplying by link count
        mass_props=sim_utils.MassPropertiesCfg(mass=DRONE_LINK_MASS),

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            fix_root_link=False,
        ),
    )

    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    )

    actuators = {
        "rotors": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=1.0e5,
            velocity_limit=1.0e5,
            stiffness=0.0,
            damping=0.0,
        )
    }


CF2X_CFG = Cf2xDroneCfg()
CRAZYFLIE_CFG = CF2X_CFG
DRONE_CFG = CF2X_CFG
