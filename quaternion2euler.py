import numpy as np
import math

def quaternion2euler (x, y, z, w):
    """
    Converts a quaternion into Euler angles (yaw, pitch, and roll).
    It handles singularities at the north and south poles.
    
    Args:
        x (float): The x component of the quaternion.
        y (float): The y component of the quaternion.
        z (float): The z component of the quaternion.
        w (float): The w component of the quaternion.
    
    Returns:
        numpy.ndarray: A 3-element array containing the Euler angles in degrees [yaw, pitch, roll].
        Notes:
        - The function handles singularities at the north and south poles.
        - The input quaternion does not need to be normalized.
    """

    def NormalizeAngle (angle):
        while (angle>2*np.pi):
            angle -= 2*np.pi
        while (angle<-0.001):
            angle += 2*np.pi
        return angle
    
    def NormalizeAngles (angles):
        angles[0] = NormalizeAngle (angles[0])
        angles[1] = NormalizeAngle (angles[1])
        angles[2] = NormalizeAngle (angles[2])
        return angles    

    sqw = w * w
    sqx = x * x
    sqy = y * y
    sqz = z * z
    unit = sqx + sqy + sqz + sqw # if normalised is one, otherwise is correction factor
    test = x * w - y * z
    v = np.zeros(3)  # array for Euler angles


    if test>0.4995*unit: # singularity at north pole
        v[1] = 2 * math.atan2(y, x)
        v[0] = math.pi / 2
        v[2] = 0
        return NormalizeAngles(np.rad2deg(v))
    
    if test<-0.4995*unit: # singularity at south pole
        v[1] = -2 * math.atan2(y, x)
        v[0] = -math.pi / 2
        v[2] = 0
        return NormalizeAngles(np.rad2deg(v))
    
    x_new = w
    y_new = z
    z_new = x
    w_new = y
    v[1] = math.atan2 (2 * x_new * w_new + 2 * y_new * z_new, 1 - 2 * (z_new * z_new + w_new * w_new))     # Yaw
    v[0] = math.asin (2 * (x_new * z_new - w_new * y_new))                             # Pitch
    v[2] = math.atan2 (2 * x_new * y_new + 2 * z_new * w_new, 1 - 2 * (y_new * y_new + z_new * z_new))      # Roll
    return NormalizeAngles (v)



