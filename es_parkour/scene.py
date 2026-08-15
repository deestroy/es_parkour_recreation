"""Builds the MJCF scene: Go2 + heightfield terrain placeholder."""
from __future__ import annotations

import os

_SCENE_TEMPLATE = """
<mujoco model="go2_parkour">
  <include file="{go2_xml}"/>
  <option timestep="{dt}"/>
  <asset>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{rx} {ry} 1.0 0.5"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 10" dir="0 0 -1"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="{cx} {cy} 0"
          friction="1.0 0.005 0.0001" priority="1"/>
  </worldbody>
</mujoco>
"""


def build_scene_xml(assets_dir: str, nrow: int, ncol: int,
                    course_length: float, course_width: float, dt: float) -> str:
    """Writes the scene XML next to go2.xml and returns its path."""
    xml = _SCENE_TEMPLATE.format(
        go2_xml="go2.xml",
        dt=dt,
        nrow=nrow, ncol=ncol,
        rx=course_length / 2.0, ry=course_width / 2.0,
        cx=course_length / 2.0, cy=0.0,
    )
    path = os.path.join(assets_dir, "scene_parkour.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path
