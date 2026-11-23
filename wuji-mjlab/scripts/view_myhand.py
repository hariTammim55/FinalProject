from pathlib import Path

import mujoco
import mujoco.viewer

# Paths resolved relative to the repo root, so the script works from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "src" / "wuji_mjlab" / "assets"
XML_PATH = ASSETS / "robots" / "MyHand" / "mjcf" / "hand_assembly.xml"
CUBE_XML_PATH = ASSETS / "objects" / "merah_putih_cube" / "xmls" / "cube.xml"
CUBE_POS = (-0.01, -0.08, 0.115)  # just above the palm (palm top z ≈ 0.077)

hand_spec = mujoco.MjSpec.from_file(str(XML_PATH))
cube_spec = mujoco.MjSpec.from_file(str(CUBE_XML_PATH))
cube_spec.body("cube").pos = (0.0, 0.0, 0.0)
# Rotate 90 deg about world Y so the two diagonally split (red/white) faces
# (cube local +/-Z) face sideways (world +/-X); top and bottom become solid faces.
# cube_spec.body("cube").quat = (0.7071068, 0.0, 0.7071068, 0.0)
cube_spec.body("cube").quat = (1.0, 0.0, 0.0, 0.0)
cube_frame = hand_spec.worldbody.add_frame(name="object_frame", pos=CUBE_POS)
hand_spec.attach(cube_spec, prefix="object/", frame=cube_frame)

model = hand_spec.compile()
data = mujoco.MjData(model)

print("Model tangan + cube merah putih berhasil dimuat.")
print("Jumlah body:", model.nbody)
print("Jumlah joint:", model.njnt)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Show the cube's visual geom (group 2) with its red/white texture.
    viewer.opt.geomgroup[2] = 1
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
