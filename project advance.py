# %%
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# =========================================================
# CONNECT TO COPPELIASIM
# =========================================================
client = RemoteAPIClient()
sim = client.require('sim')

# =========================================================
# START SIMULATION
# =========================================================
sim.startSimulation()
print("Simulation Started")

# =========================================================
# HOMOGENEOUS TRANSFORMATION MATRIX
# =========================================================
def transformMat(alpha, beta, gamma, tx, ty, tz):

    # Rotation X
    rotx = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha),  math.cos(alpha)]
    ])

    # Rotation Y
    roty = np.array([
        [ math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)]
    ])

    # Rotation Z
    rotz = np.array([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma),  math.cos(gamma), 0],
        [0,0,1]
    ])

    # Total rotation matrix
    rot_total = rotx @ roty @ rotz

    # Translation vector
    trans_vector = np.array([
        [tx],
        [ty],
        [tz]
    ])

    # Combine rotation and translation
    Rt = np.hstack((rot_total, trans_vector))

    # Homogeneous row
    homogeneous_row = np.array([[0, 0, 0, 1]])

    # Final transformation matrix
    T = np.vstack((Rt, homogeneous_row))

    return T

# =========================================================
# GET OBJECT HANDLES
# =========================================================
sim.addLog(1, "Python Connected!")

# Robot
p3dx = sim.getObject("/PioneerP3DX")

# Motors
p3dx_rw = sim.getObject("/PioneerP3DX/rightMotor")
p3dx_lw = sim.getObject("/PioneerP3DX/leftMotor")

# Visualization objects
LH_Handle = sim.getObject("/LH")
perp_Handle = sim.getObject("/Perp")

# =========================================================
# PATH WAYPOINTS
# =========================================================
path_Handle = []

# Total waypoint = 62
for i in range(0, 62):

    path_Handle.append(
        sim.getObject(f"/p[{i}]")
    )

# =========================================================
# ULTRASONIC SENSORS
# =========================================================
sensor_handles = []

# Use multiple sensors for denser environment mapping
for i in range(1, 16):

    sensor_handles.append(
        sim.getObject(
            f"/PioneerP3DX/ultrasonicSensor[{i}]"
        )
    )

# =========================================================
# ROBOT PARAMETERS
# =========================================================
rw = 0.195 / 2
rb = 0.318 / 2

# Look-ahead distance
LH_distance = 0.6

# =========================================================
# DATA STORAGE
# =========================================================
x_odom = []
y_odom = []

map_x = []
map_y = []

# =========================================================
# MAIN LOOP
# =========================================================
try:

    start_time = time.time()
    elapsed_prev = 0.0

    # Run simulation for 45 seconds
    while (time.time() - start_time) < 100:

        # =================================================
        # TIME UPDATE
        # =================================================
        elapsed = time.time()

        dt = elapsed - elapsed_prev
        elapsed_prev = elapsed

        # =================================================
        # ROBOT POSE
        # =================================================
        p3dx_position = sim.getObjectPosition(
            p3dx,
            sim.handle_world
        )

        p3dx_orientation = sim.getObjectOrientation(
            p3dx,
            sim.handle_world
        )

        # Save trajectory
        x_odom.append(p3dx_position[0])
        y_odom.append(p3dx_position[1])

        # =================================================
        # SENSOR MAPPING
        # =================================================
        for sensor in sensor_handles:

            result, dist, point, obj, n = sim.readProximitySensor(sensor)

            if result > 0:

                # Sensor transformation matrix
                sensor_matrix = sim.getObjectMatrix(
                    sensor,
                    sim.handle_world
                )

                px, py, pz = point

                # Convert local sensor coordinate to world coordinate
                world_x = (
                    sensor_matrix[0] * px
                    +
                    sensor_matrix[1] * py
                    +
                    sensor_matrix[2] * pz
                    +
                    sensor_matrix[3]
                )

                world_y = (
                    sensor_matrix[4] * px
                    +
                    sensor_matrix[5] * py
                    +
                    sensor_matrix[6] * pz
                    +
                    sensor_matrix[7]
                )

                map_x.append(world_x)
                map_y.append(world_y)

        # =================================================
        # LOOK AHEAD POSITION
        # =================================================
        LH_position_to_world = (
            transformMat(
                0,
                0,
                p3dx_orientation[2],
                p3dx_position[0],
                p3dx_position[1],
                p3dx_position[2]
            )
            @
            np.array([
                [LH_distance],
                [0],
                [0],
                [1]
            ])
        )

        # Remove homogeneous coordinate
        LH_position_to_world = LH_position_to_world[:3, :]

        # =================================================
        # GET WAYPOINT POSITIONS
        # =================================================
        path_points = []

        for i in range(len(path_Handle)):

            point_position = sim.getObjectPosition(
                path_Handle[i],
                sim.handle_world
            )

            path_points.append(point_position)

        # =================================================
        # CREATE PATH SEGMENT VECTORS
        # =================================================
        vec_AB = []

        for i in range(len(path_points)-1):

            A = np.array(path_points[i]).reshape(3,1)
            B = np.array(path_points[i+1]).reshape(3,1)

            vec_AB.append(B - A)

        # Close loop path
        A = np.array(path_points[-1]).reshape(3,1)
        B = np.array(path_points[0]).reshape(3,1)

        vec_AB.append(B - A)

        # =================================================
        # VECTOR FROM PATH TO LOOK-AHEAD POINT
        # =================================================
        vec_ALH = []

        for i in range(len(path_points)):

            A = np.array(path_points[i]).reshape(3,1)

            vec_ALH.append(
                LH_position_to_world - A
            )

        # =================================================
        # PROJECTION POINT CALCULATION
        # =================================================
        scalar_proj_points = []

        for i in range(len(vec_AB)):

            AB = vec_AB[i]
            ALH = vec_ALH[i]

            numerator = np.dot(ALH.T, AB).item()
            denominator = np.dot(AB.T, AB).item()

            # Avoid divide by zero
            if denominator == 0:

                scalar_proj = 0

            else:

                scalar_proj = numerator / denominator

            # Clamp projection value
            scalar_proj = max(0, min(1, scalar_proj))

            A = np.array(path_points[i]).reshape(3,1)

            projection_point = A + scalar_proj * AB

            scalar_proj_points.append(projection_point)

        # =================================================
        # FIND NEAREST PROJECTION POINT
        # =================================================
        closest_index = 0
        min_distance = float('inf')

        for i in range(len(scalar_proj_points)):

            dx = (
                scalar_proj_points[i][0,0]
                -
                LH_position_to_world[0,0]
            )

            dy = (
                scalar_proj_points[i][1,0]
                -
                LH_position_to_world[1,0]
            )

            distance = math.sqrt(dx**2 + dy**2)

            if distance < min_distance:

                min_distance = distance
                closest_index = i

        # =================================================
        # DESIRED TRACKING POSITION
        # =================================================
        desired_position = scalar_proj_points[closest_index]

        # =================================================
        # TRANSFORM TO ROBOT FRAME
        # =================================================
        T_world_robot = transformMat(
            0,
            0,
            p3dx_orientation[2],
            p3dx_position[0],
            p3dx_position[1],
            p3dx_position[2]
        )

        desired_position_wrt_robot = (
            np.linalg.inv(T_world_robot)
            @
            np.append(
                desired_position,
                np.array([[1]]),
                axis=0
            )
        )

        desired_position_wrt_robot = desired_position_wrt_robot[:3, :]

        # =================================================
        # TRACKING ERROR
        # =================================================
        ed = math.sqrt(
            desired_position_wrt_robot[0,0]**2
            +
            desired_position_wrt_robot[1,0]**2
        )

        eh = math.atan2(
            desired_position_wrt_robot[1,0],
            desired_position_wrt_robot[0,0]
        )

        # =================================================
        # MOTION CONTROLLER
        # =================================================
        # Reduce forward speed while turning
        vx = max(0.2 - 0.1 * abs(eh), 0.05)

        # Angular controller
        wx = 0.5 * eh

        # Limit turning speed
        wx = max(min(wx, 0.8), -0.8)

        # =================================================
        # DIFFERENTIAL DRIVE KINEMATICS
        # =================================================
        wr_vel = (vx + (rb * wx)) / rw
        wl_vel = (vx - (rb * wx)) / rw

        # =================================================
        # SEND MOTOR COMMANDS
        # =================================================
        sim.setJointTargetVelocity(
            p3dx_rw,
            wr_vel
        )

        sim.setJointTargetVelocity(
            p3dx_lw,
            wl_vel
        )

        # =================================================
        # VISUALIZATION
        # =================================================
        sim.setObjectPosition(
            LH_Handle,
            sim.handle_world,
            LH_position_to_world.flatten().tolist()
        )

        sim.setObjectPosition(
            perp_Handle,
            sim.handle_world,
            desired_position.flatten().tolist()
        )

finally:

    # =====================================================
    # STOP ROBOT SAFELY
    # =====================================================
    sim.setJointTargetVelocity(p3dx_rw, 0)
    sim.setJointTargetVelocity(p3dx_lw, 0)

    sim.stopSimulation()

    print("\nSimulation Stopped")

# =========================================================
# ENVIRONMENT MAP VISUALIZATION
# =========================================================
plt.figure(figsize=(10,10))

# Obstacle mapping
plt.scatter(
    map_x,
    map_y,
    s=2,
    c='red',
    label='Sensor Mapping'
)

# Robot trajectory
plt.plot(
    x_odom,
    y_odom,
    c='blue',
    linewidth=2,
    label='Robot Trajectory'
)

plt.title("Environment Mapping and Path Tracking")

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.legend()

plt.grid(True)
plt.axis('equal')

plt.show()

# %%