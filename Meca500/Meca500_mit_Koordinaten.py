###############################################################
# vorher muss mecademicpy und numpy über pip installiert sein #
###############################################################
import time
import mecademicpy.robot as mdr
from mecademicpy.robot_classes import CommunicationError
import numpy as np

########################################
# Hilfsfunktionen für Transformationen #
########################################

def rotx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]])

def roty(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s,0, c]])

def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]])

def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

###########################
# TCP relativ zum Flansch #
###########################
TCP_OFFSET = np.array([0, 0, 0.036])     # 36 mm
TCP_ROT = rotz(np.deg2rad(45))           # 45° um Z
T_FCP_to_TCP = make_transform(TCP_ROT, TCP_OFFSET)
#Blick richtung der Kamera ist Z-Achse des Tools


#########################
# Roboter-Konfiguration #
#########################
ROBOT_IP = "192.168.0.100"
ROBOT_VEL = 20
ROBOT_ACC = 20

POSITIONS = {
    "links":  (48, 14, 24, -108, 91, -81),
    "mitte":  (0, -35, 35, 0, 0, 135),
    "rechts": (-41, 14, 24, 106, 89, 6)
}

HOME_JOINTS = (0, 0, 0, 0, 0, 0)

#######################
# Robotervorbereitung #
#######################
def ensure_robot_ready(robot):
    try: robot.ResetError()
    except: pass
    try: robot.ClearMotion()
    except: pass
    robot.ActivateAndHome()
    robot.WaitHomed()

#################
# HAUPTPROGRAMM #
#################
def main():
    robot = mdr.Robot()

    try:
        print("[ROBOT] Verbinde ...")
        robot.Connect(ROBOT_IP)
        print("[ROBOT] Verbunden.")

        ensure_robot_ready(robot)
        robot.SetJointVel(ROBOT_VEL)
        robot.SetJointAcc(ROBOT_ACC)

        print("Gib die gewünschte Position ein (links, mitte, rechts).")
        print("Tippe 'exit', um das Programm zu beenden.")

        while True:
            eingabe = input("Position: ").lower()

            if eingabe == "exit":
                break
            if eingabe not in POSITIONS:
                print(f"[FEHLER] Ungültige Eingabe: {eingabe}")
                continue

            print(f"[ROBOT] Fahre zu Position '{eingabe}' ...")
            robot.MoveJoints(*POSITIONS[eingabe])
            robot.WaitIdle()
            print(f"[ROBOT] Position '{eingabe}' erreicht.")

            #####################
            # TCP Pose auslesen #
            #####################
            x, y, z, rx, ry, rz = robot.GetPose()

            # Pose in mm → m umrechnen
            t_tcp_world = np.array([x, y, z]) / 1000.0

            # Euler ZYX → Rotationsmatrix
            R_tcp_world = rotz(np.deg2rad(rz)) @ \
                           roty(np.deg2rad(ry)) @ \
                           rotx(np.deg2rad(rx))

            print("\n==============================")
            print("      KAMERA / TCP POSE")
            print("==============================")

            
            # 1. Weltposition (mm)
            print("Kamera Position (WORLD):")
            print(f"   x={x:.3f} mm")
            print(f"   y={y:.3f} mm")
            print(f"   z={z:.3f} mm\n")

            
            # 2. Orientierung als Achsenvektoren
            print("Kamera Orientierung (WORLD-ACHSEN):")
            cam_x = R_tcp_world[:,0]
            cam_y = R_tcp_world[:,1]
            cam_z = R_tcp_world[:,2]

            print(f"   X-Achse: {cam_x}")
            print(f"   Y-Achse: {cam_y}")
            print(f"   Z-Achse: {cam_z}\n")

            print("==============================\n")

        print("[ROBOT] Fahre zurück zur Home-Position ...")
        robot.WaitIdle()

    except (TimeoutError, CommunicationError) as e:
        print("[ROBOT] Kommunikationsfehler:", e)

    finally:
        try:
            robot.DeactivateRobot()
            robot.WaitDeactivated()
        except: pass
        try:
            robot.Disconnect()
        except: pass

        print("[ROBOT] Roboter deaktiviert und getrennt.")

# =========================================

if __name__ == "__main__":
    main()
