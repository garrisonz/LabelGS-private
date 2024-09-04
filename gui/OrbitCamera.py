import numpy as np
import math
from scipy.spatial.transform import Rotation as R


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.update_fovy_degrees(fovy)
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0
    
    def fovy_degrees(self):
        return self.fovy * 180.0 / math.pi
    
    def update_fovy_degrees(self, fovy):
        self.fovy = fovy * math.pi / 180.0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res

    def pose_movecenter2(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        R = rot[:3, :3]

        # --- translate: tc --- #
        T = res[:3, 3]
        T -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        T = -R.transpose() @ T
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])

    def update(self, cam):
        self.update_from_transform(cam.R, cam.T, cam.FoVy)

    def update_from_transform(self, RotaionMatrix, Translation, fovy):
        self.rot = R.from_matrix(RotaionMatrix)

        T = Translation
        T = self.rot.as_matrix() @ T
        self.center = T
        self.radius = 0
        #self.fovy = fovy





