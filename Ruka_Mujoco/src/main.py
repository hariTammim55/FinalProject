import os
import time
import numpy as np

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/xml/hand_box_scene.xml")
data = mujoco.MjData(model)

# ============================================================
# JOINT RANGE (DEGREE)
# ============================================================

# Index Finger
# Index_MCP      : 0° → 140°
# Index_DIP_PIP  : 0° → 120°

# Middle Finger
# Middle_MCP     : 0° → 140°
# Middle_DIP_PIP : 0° → 120°

# Ring Finger
# Ring_MCP       : 0° → 140°
# Ring_DIP_PIP   : 0° → 120°

# Pinky Finger
# Pinky_MCP      : 0° → 140°
# Pinky_DIP_PIP  : 0° → 120°

# Thumb
# Thumb_MCP      : 0° → 90°
# Thumb_PIP      : 0° → 120°
# Thumb_DIP      : 0° → 190°

Index_MCP       = 0.0
Index_DIP_PIP   = 0.0
Middle_MCP      = 0.0
Middle_DIP_PIP  = 0.0
Ring_MCP        = 0.0
Ring_DIP_PIP    = 0.0
Pinky_MCP       = 0.0
Pinky_DIP_PIP   = 0.0
Thumb_CMC       = 0.0
Thumb_PIP       = 0.0
Thumb_DIP       = 0.0

def Move():
    data.ctrl[0]    = np.deg2rad(Index_MCP)
    data.ctrl[1]    = np.deg2rad(Index_DIP_PIP)
    data.ctrl[2]    = np.deg2rad(Index_DIP_PIP)
    data.ctrl[3]    = np.deg2rad(Middle_MCP)
    data.ctrl[4]    = np.deg2rad(Middle_DIP_PIP)
    data.ctrl[5]    = np.deg2rad(Middle_DIP_PIP)
    data.ctrl[6]    = np.deg2rad(Ring_MCP)
    data.ctrl[7]    = np.deg2rad(Ring_DIP_PIP)
    data.ctrl[8]    = np.deg2rad(Ring_DIP_PIP)
    data.ctrl[9]    = np.deg2rad(Pinky_MCP)
    data.ctrl[10]   = np.deg2rad(Pinky_DIP_PIP)
    data.ctrl[11]   = np.deg2rad(Pinky_DIP_PIP)
    data.ctrl[12]   = np.deg2rad(Thumb_CMC)
    data.ctrl[13]   = np.deg2rad(Thumb_PIP)
    data.ctrl[14]   = np.deg2rad(Thumb_DIP)

def Grasp():
    global Index_MCP, Index_DIP_PIP, Middle_MCP, Middle_DIP_PIP, Ring_MCP, Ring_DIP_PIP, Pinky_MCP, Pinky_DIP_PIP, Thumb_CMC, Thumb_PIP, Thumb_DIP

    Index_MCP       = 140.0
    Index_DIP_PIP   = 120.0
    Middle_MCP      = 140.0
    Middle_DIP_PIP  = 120.0
    Ring_MCP        = 140.0
    Ring_DIP_PIP    = 120.0
    Pinky_MCP       = 140.0
    Pinky_DIP_PIP   = 120.0
    Thumb_CMC       = 90.0
    Thumb_PIP       = 120.0
    Thumb_DIP       = 190.0

    Move()

def Release():
    global Index_MCP, Index_DIP_PIP, Middle_MCP, Middle_DIP_PIP, Ring_MCP, Ring_DIP_PIP, Pinky_MCP, Pinky_DIP_PIP, Thumb_CMC, Thumb_PIP, Thumb_DIP

    Index_MCP       = 0.0
    Index_DIP_PIP   = 0.0
    Middle_MCP      = 0.0
    Middle_DIP_PIP  = 0.0
    Ring_MCP        = 0.0
    Ring_DIP_PIP    = 0.0
    Pinky_MCP       = 0.0
    Pinky_DIP_PIP   = 0.0
    Thumb_CMC       = 0.0
    Thumb_PIP       = 0.0
    Thumb_DIP       = 0.0

    Move()

def Hand_1():

    global Index_MCP, Index_DIP_PIP
    global Middle_MCP, Middle_DIP_PIP
    global Ring_MCP, Ring_DIP_PIP
    global Pinky_MCP, Pinky_DIP_PIP
    global Thumb_CMC, Thumb_PIP, Thumb_DIP

    # =========================
    # ANGKA 1
    # =========================

    # Index = OPEN
    Index_MCP = 0.0
    Index_DIP_PIP = 0.0

    # Middle = CLOSED
    Middle_MCP = 140.0
    Middle_DIP_PIP = 120.0

    # Ring = CLOSED
    Ring_MCP = 140.0
    Ring_DIP_PIP = 120.0

    # Pinky = CLOSED
    Pinky_MCP = 140.0
    Pinky_DIP_PIP = 120.0

    # Thumb = CLOSED
    Thumb_CMC = 90.0
    Thumb_PIP = 120.0
    Thumb_DIP = 190.0

    Move()


def Hand_2():

    global Index_MCP, Index_DIP_PIP
    global Middle_MCP, Middle_DIP_PIP
    global Ring_MCP, Ring_DIP_PIP
    global Pinky_MCP, Pinky_DIP_PIP
    global Thumb_CMC, Thumb_PIP, Thumb_DIP

    # =========================
    # ANGKA 2
    # =========================

    # Index = OPEN
    Index_MCP = 0.0
    Index_DIP_PIP = 0.0

    # Middle = OPEN
    Middle_MCP = 0.0
    Middle_DIP_PIP = 0.0

    # Ring = CLOSED
    Ring_MCP = 140.0
    Ring_DIP_PIP = 120.0

    # Pinky = CLOSED
    Pinky_MCP = 140.0
    Pinky_DIP_PIP = 120.0

    # Thumb = CLOSED
    Thumb_CMC = 90.0
    Thumb_PIP = 120.0
    Thumb_DIP = 190.0

    Move()


def Hand_3():

    global Index_MCP, Index_DIP_PIP
    global Middle_MCP, Middle_DIP_PIP
    global Ring_MCP, Ring_DIP_PIP
    global Pinky_MCP, Pinky_DIP_PIP
    global Thumb_CMC, Thumb_PIP, Thumb_DIP

    # =========================
    # ANGKA 3
    # =========================

    # Index = OPEN
    Index_MCP = 0.0
    Index_DIP_PIP = 0.0

    # Middle = OPEN
    Middle_MCP = 0.0
    Middle_DIP_PIP = 0.0

    # Ring = OPEN
    Ring_MCP = 0.0
    Ring_DIP_PIP = 0.0

    # Pinky = CLOSED
    Pinky_MCP = 140.0
    Pinky_DIP_PIP = 120.0

    # Thumb = CLOSED
    Thumb_CMC = 90.0
    Thumb_PIP = 120.0
    Thumb_DIP = 190.0

    Move()


def Hand_4():

    global Index_MCP, Index_DIP_PIP
    global Middle_MCP, Middle_DIP_PIP
    global Ring_MCP, Ring_DIP_PIP
    global Pinky_MCP, Pinky_DIP_PIP
    global Thumb_CMC, Thumb_PIP, Thumb_DIP

    # =========================
    # ANGKA 4
    # =========================

    # Index = OPEN
    Index_MCP = 0.0
    Index_DIP_PIP = 0.0

    # Middle = OPEN
    Middle_MCP = 0.0
    Middle_DIP_PIP = 0.0

    # Ring = OPEN
    Ring_MCP = 0.0
    Ring_DIP_PIP = 0.0

    # Pinky = OPEN
    Pinky_MCP = 0.0
    Pinky_DIP_PIP = 0.0

    # Thumb = CLOSED
    Thumb_CMC = 90.0
    Thumb_PIP = 120.0
    Thumb_DIP = 190.0

    Move()


def Hand_5():

    global Index_MCP, Index_DIP_PIP
    global Middle_MCP, Middle_DIP_PIP
    global Ring_MCP, Ring_DIP_PIP
    global Pinky_MCP, Pinky_DIP_PIP
    global Thumb_CMC, Thumb_PIP, Thumb_DIP

    # =========================
    # ANGKA 5
    # =========================

    # Semua jari OPEN

    Index_MCP = 0.0
    Index_DIP_PIP = 0.0

    Middle_MCP = 0.0
    Middle_DIP_PIP = 0.0

    Ring_MCP = 0.0
    Ring_DIP_PIP = 0.0

    Pinky_MCP = 0.0
    Pinky_DIP_PIP = 0.0

    # Thumb OPEN
    Thumb_CMC = 0.0
    Thumb_PIP = 0.0
    Thumb_DIP = 0.0

    Move()

# =========================
# LAUNCH MUJOCO VIEWER
# =========================
start_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.opt.geomgroup[2] = False

    while viewer.is_running():

        

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)