from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required to load iris.yaml parameters for omniperception_isaacdrone."
    ) from exc

_THIS_DIR = Path(__file__).resolve().parent
_IRIS_USD_PATH = str(_THIS_DIR / "iris.usd")
_IRIS_PARAM_PATH = _THIS_DIR / "iris.yaml"

with _IRIS_PARAM_PATH.open("r", encoding="utf-8") as f:
    IRIS_PARAMS = yaml.safe_load(f)

DRONE_NAME: str = str(IRIS_PARAMS.get("name", "iris"))
DRONE_MASS: float = float(IRIS_PARAMS["mass"])
DRONE_INERTIA_DIAG: tuple[float, float, float] = (
    float(IRIS_PARAMS["inertia"]["xx"]),
    float(IRIS_PARAMS["inertia"]["yy"]),
    float(IRIS_PARAMS["inertia"]["zz"]),
)


@configclass
class IrisDroneCfg(ArticulationCfg):
    """Iris drone articulation config.

    Note:
        We do NOT overwrite mass_props here. The Iris USD already contains per-link
        physical parameters; overriding mass_props at articulation level can duplicate
        mass across links and break total mass.
    """

    spawn = sim_utils.UsdFileCfg(
        usd_path=_IRIS_USD_PATH,
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
            joint_names_expr=["rotor_.*"],
            effort_limit_sim=1.0e5,
            velocity_limit_sim=1.0e5,
            stiffness=0.0,
            damping=0.0,
        )
    }


IRIS_CFG = IrisDroneCfg()
DRONE_CFG = IRIS_CFG
DRONE_PARAMS = IRIS_PARAMS
