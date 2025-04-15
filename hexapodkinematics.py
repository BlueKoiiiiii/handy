import math

def inverse_kinematics(x, y, z, leg, side):
    Y_rest = 8
    Z_rest = -9
    """Calculate inverse kinematics for the hexapod leg."""
    COXA = 5   # Length of the COXA segment
    FEMUR = 6 # Length of the FEMUR segment
    TIBIA = 8.5  # Length of the TIBIA segment
    # print(leg, side, x, y, z)
    if side == "right":
        x*=-1
        if leg == "front":
            x -= 10
        elif leg == "back":
            x += 10
        y += Y_rest

    if side == "left": 
        y*=-1
        y += Y_rest
        if leg == "front":
            x += 10
        elif leg == "back":
            x -= 10
   
    z += Z_rest

    J1 = math.atan2(x,y)  # Fixed argument order
    H = math.sqrt(x**2 + y**2)
    L = math.sqrt((H - COXA)**2 + z**2)
    J3 = math.acos((FEMUR**2 + TIBIA**2 - L**2) / (2 * FEMUR * TIBIA))
    B = math.acos((FEMUR**2 + L**2 - TIBIA**2) / (2 * FEMUR * L))
    A = math.atan2(z, (H - COXA))
    J2 = B + A
    print(leg, side, x, y, z)
    return math.degrees(J1), math.degrees(J2), math.degrees(J3)

def arminverse_kinematics(z, y, theta, elbow_up=True):    
    # Robot arm link lengths
    length1 = 60
    length2 = 150
    
    # Distance from the base to the target point
    r = math.sqrt(z**2 + y**2)
    
    # Elbow angle (theta2)
    cos_theta2 = (length1**2 + length2**2 - r**2) / (2 * length1 * length2)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    
    if not elbow_up:
        theta2 = -theta2
    
    # Shoulder angle (theta1)
    cos_theta1 = (r**2 + length1**2 - length2**2) / (2 * r * length1)
    cos_theta1 = max(min(cos_theta1, 1), -1)
    theta1_offset = math.acos(cos_theta1)
    base_angle = math.atan2(y, z)
    
    theta1_rad = base_angle + theta1_offset if elbow_up else base_angle - theta1_offset
    
    theta1 = math.degrees(theta1_rad)
    theta2 = math.degrees(theta2)
    
    # Apply joint limits
    theta1 = max(min(theta1, 180), 0)
    theta2 = max(min(theta2, 180), 0)
    
    # Wrist compensation angle
    compensation_angle = theta1 + theta2
    theta3 = 270 + theta - compensation_angle - 90
    theta3 = max(min(theta3, 180), 0)

    return theta1, theta2, theta3