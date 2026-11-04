
from pathlib import Path

import mujoco
import mujoco.viewer

XML_PATH = (
    "src/wuji_mjlab/assets/robots/myHand/"
    "mjcf/hand_assembly.xml"
)
CUBE_XML_PATH = "src/wuji_mjlab/assets/objects/inhand_object/xmls/cube.xml"
CUBE_POS = (-0.0, -0.04, 0.58)

hand_spec = mujoco.MjSpec.from_file(str(Path(XML_PATH)))
cube_spec = mujoco.MjSpec.from_file(str(Path(CUBE_XML_PATH)))
cube_spec.body("cube").pos = (0.0, 0.0, 0.0)
cube_frame = hand_spec.worldbody.add_frame(name="object_frame", pos=CUBE_POS)
hand_spec.attach(cube_spec, prefix="object/", frame=cube_frame)

model = hand_spec.compile()
data = mujoco.MjData(model)

print("Model tangan berhasil dimuat.")
print("Jumlah body:", model.nbody)
print("Jumlah joint:", model.njnt)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()


