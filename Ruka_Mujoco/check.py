
import mujoco

# Muat model scene tangan dan kubus
model = mujoco.MjModel.from_xml_path("assets/xml/hand_box_scene.xml")
data = mujoco.MjData(model)

# Jalankan simulasi selama 10 detik
for _ in range(5000):
    mujoco.mj_step(model, data)

# Tampilkan semua kontak yang terdeteksi
print("Jumlah kontak:", data.ncon)
print("-" * 60)

for i in range(data.ncon):
    c = data.contact[i]

    g1 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1
    )
    g2 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2
    )

    if {g1, g2} == {"cube", "Palm_c"}:
        print("Pasangan:", g1, "<->", g2)
        print("Jarak:", c.dist, "m")
        print("Posisi kontak:", c.pos)
        print("Margin kontak:", c.includemargin)
        print("Normal kontak:", c.frame[:3])
        print("-" * 40)
