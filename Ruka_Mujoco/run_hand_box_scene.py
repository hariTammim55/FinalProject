
import mujoco
import mujoco.viewer
from pathlib import Path
import time

# Path XML berdasarkan lokasi file Python ini
BASE_DIR = Path(__file__).resolve().parent
XML_PATH = BASE_DIR / "assets" / "xml" / "hand_box_scene.xml"


def main():
    # Periksa keberadaan file XML
    if not XML_PATH.is_file():
        raise FileNotFoundError(
            f"File XML tidak ditemukan: {XML_PATH}"
        )

    # Muat model dan data simulasi
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Jalankan viewer interaktif
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()

            # Jalankan satu langkah simulasi
            mujoco.mj_step(model, data)

            # Sinkronkan viewer
            viewer.sync()

            # Batasi kecepatan loop mendekati waktu simulasi
            elapsed = time.time() - start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
