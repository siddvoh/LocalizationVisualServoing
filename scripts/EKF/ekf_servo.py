# EKF for object pose in camera frame, plus PBVS controller + depth scaler.
# state = [X, Y, Z, Vx, Vy, Vz]; meas = (u, v, Z) from centroid + mono depth.

import numpy as np


class CameraIntrinsics:
    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def update(self, fx, fy, cx, cy):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def project(self, X, Y, Z):
        Z = max(Z, 1e-4)
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        return u, v

    def backproject(self, u, v, Z):
        Z = max(Z, 1e-4)
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return X, Y, Z


class PoseEKF:
    """EKF over [X, Y, Z, Vx, Vy, Vz] in camera frame."""

    DIM_STATE = 6
    DIM_MEAS = 3  # (u, v, Z)

    def __init__(self, cam: CameraIntrinsics,
                 process_noise_pos=0.005,
                 process_noise_vel=0.05,
                 meas_noise_uv=8.0,
                 meas_noise_z=0.15):
        self.cam = cam

        self.x = np.zeros(self.DIM_STATE)
        self.P = np.eye(self.DIM_STATE)

        self.Q = np.diag([
            process_noise_pos ** 2,
            process_noise_pos ** 2,
            process_noise_pos ** 2,
            process_noise_vel ** 2,
            process_noise_vel ** 2,
            process_noise_vel ** 2,
        ])

        self.R = np.diag([
            meas_noise_uv ** 2,
            meas_noise_uv ** 2,
            meas_noise_z ** 2,
        ])

        self._initialised = False

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:6].copy()

    @property
    def is_initialised(self):
        return self._initialised

    def reset(self):
        self.x = np.zeros(self.DIM_STATE)
        self.P = np.eye(self.DIM_STATE)
        self._initialised = False

    def initialise(self, u, v, Z):
        X, Y, Z = self.cam.backproject(u, v, Z)
        self.x = np.array([X, Y, Z, 0.0, 0.0, 0.0])
        self.P = np.diag([0.05, 0.05, 0.1, 0.5, 0.5, 0.5])
        self._initialised = True

    def predict(self, dt, robot_vel_cam=None):
        """constant-velocity predict with optional egomotion compensation"""
        if not self._initialised:
            return

        F = np.eye(self.DIM_STATE)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.x = F @ self.x

        # egomotion: object shifts opposite to camera motion
        if robot_vel_cam is not None:
            v_cam = np.asarray(robot_vel_cam, dtype=np.float64)
            self.x[0] -= v_cam[0] * dt
            self.x[1] -= v_cam[1] * dt
            self.x[2] -= v_cam[2] * dt

        Q_scaled = self.Q * dt
        self.P = F @ self.P @ F.T + Q_scaled

        if self.x[2] < 0.05:
            self.x[2] = 0.05

    def update(self, u, v, Z):
        if not self._initialised:
            self.initialise(u, v, Z)
            return

        X, Y, Zs = self.x[0], self.x[1], max(self.x[2], 1e-4)

        u_pred = self.cam.fx * X / Zs + self.cam.cx
        v_pred = self.cam.fy * Y / Zs + self.cam.cy
        z_pred = np.array([u_pred, v_pred, Zs])

        # H = dh/dx, pinhole jacobian
        H = np.zeros((self.DIM_MEAS, self.DIM_STATE))
        H[0, 0] = self.cam.fx / Zs
        H[0, 2] = -self.cam.fx * X / (Zs ** 2)
        H[1, 1] = self.cam.fy / Zs
        H[1, 2] = -self.cam.fy * Y / (Zs ** 2)
        H[2, 2] = 1.0

        z_meas = np.array([u, v, Z])
        y = z_meas - z_pred

        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(self.DIM_STATE) - K @ H
        # joseph form
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        if self.x[2] < 0.05:
            self.x[2] = 0.05

    def update_3d_position(self, X, Y, Z, meas_noise_pos=0.01):
        """direct 3D position update (e.g. from FoundationPose)"""
        if not self._initialised:
            self.x = np.array([X, Y, Z, 0.0, 0.0, 0.0])
            self.P = np.diag([0.02, 0.02, 0.05, 0.5, 0.5, 0.5])
            self._initialised = True
            return

        H = np.zeros((3, self.DIM_STATE))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R3 = np.eye(3) * (meas_noise_pos ** 2)

        z_meas = np.array([X, Y, Z])
        z_pred = self.x[:3].copy()
        y = z_meas - z_pred

        S = H @ self.P @ H.T + R3
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(self.DIM_STATE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R3 @ K.T

        if self.x[2] < 0.05:
            self.x[2] = 0.05

    def update_2d_only(self, u, v):
        """pixel-only update, skips depth"""
        if not self._initialised:
            return

        X, Y, Zs = self.x[0], self.x[1], max(self.x[2], 1e-4)

        u_pred = self.cam.fx * X / Zs + self.cam.cx
        v_pred = self.cam.fy * Y / Zs + self.cam.cy
        z_pred = np.array([u_pred, v_pred])

        H = np.zeros((2, self.DIM_STATE))
        H[0, 0] = self.cam.fx / Zs
        H[0, 2] = -self.cam.fx * X / (Zs ** 2)
        H[1, 1] = self.cam.fy / Zs
        H[1, 2] = -self.cam.fy * Y / (Zs ** 2)

        R_2d = self.R[:2, :2]
        z_meas = np.array([u, v])
        y = z_meas - z_pred

        S = H @ self.P @ H.T + R_2d
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(self.DIM_STATE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_2d @ K.T

        if self.x[2] < 0.05:
            self.x[2] = 0.05

    def predicted_pixel(self):
        if not self._initialised:
            return None
        X, Y, Z = self.x[0], self.x[1], max(self.x[2], 1e-4)
        u, v = self.cam.project(X, Y, Z)
        return int(round(u)), int(round(v))

    def position_covariance(self):
        return self.P[:3, :3].copy()

    def position_uncertainty(self):
        return float(np.sqrt(np.trace(self.P[:3, :3])))


class PBVSController:
    """PBVS, eye-in-hand. v_cam = +lambda*(t_obj - t_desired) then rotate to robot."""

    def __init__(self,
                 gain=0.5,
                 target_depth=0.30,
                 max_vel=0.02,
                 dead_zone_m=0.005,
                 R_cam_to_robot=None):
        self.gain = gain
        self.target_depth = target_depth
        self.max_vel = max_vel
        self.dead_zone_m = dead_zone_m

        if R_cam_to_robot is None:
            self.R_cam_to_robot = np.eye(3)
        else:
            self.R_cam_to_robot = np.asarray(R_cam_to_robot, dtype=np.float64)

    def compute_velocity(self, obj_position_cam):
        """returns (v_robot, v_cam, err_norm) in m/s, m/s, m"""
        desired = np.array([0.0, 0.0, self.target_depth])
        error = obj_position_cam - desired
        err_norm = float(np.linalg.norm(error))

        if err_norm < self.dead_zone_m:
            return np.zeros(3), np.zeros(3), err_norm

        # world-static target: v_cam points in the same direction as error
        v_cam = self.gain * error

        v_cam = np.clip(v_cam, -self.max_vel, self.max_vel)

        v_robot = self.R_cam_to_robot @ v_cam

        return v_robot, v_cam, err_norm

    def robot_vel_to_cam(self, v_robot):
        return self.R_cam_to_robot.T @ np.asarray(v_robot, dtype=np.float64)


class DepthScaler:
    """affine map: Z_metric = scale * d_relative + offset"""

    def __init__(self, default_scale=0.001, default_offset=0.0):
        self.scale = default_scale
        self.offset = default_offset
        self.calibrated = False

    def calibrate(self, d_relative, Z_metric_known):
        """one-point calibration, assumes offset=0"""
        if d_relative > 1e-6:
            self.scale = Z_metric_known / d_relative
            self.offset = 0.0
            self.calibrated = True

    def to_metric(self, d_relative):
        return self.scale * d_relative + self.offset

    def estimate_object_depth(self, depth_map_relative, mask, percentile=50):
        m = mask > 0
        if not np.any(m):
            return None
        d_rel = float(np.percentile(depth_map_relative[m], percentile))
        return self.to_metric(d_rel)
