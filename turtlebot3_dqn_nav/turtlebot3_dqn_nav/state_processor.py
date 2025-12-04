# state_processor.py
import numpy as np

class StateProcessor:
    """
    Procesa LiDAR + info de goal en un vector de estado:
    [ n_bins de LiDAR , distancia_normalizada_goal , ángulo_relativo_normalizado ]
    """

    def __init__(self, n_lidar_bins: int = 10, max_lidar_range: float = 3.5):
        """
        n_lidar_bins: número de sectores en los que se agrupa el LiDAR (360°)
        max_lidar_range: rango máximo del LiDAR (TurtleBot3 ~3.5m)
        """
        self.n_lidar_bins = n_lidar_bins
        self.max_lidar_range = max_lidar_range

    def process_lidar(self, scan_data):
        """Convierte 360 lecturas en n_lidar_bins (mínimo por sector, normalizado)."""
        scan_array = np.array(scan_data, dtype=np.float32)

        # Reemplazar inf/NaN por rango máximo
        scan_array[np.isinf(scan_array)] = self.max_lidar_range
        scan_array[np.isnan(scan_array)] = self.max_lidar_range

        # Clipping
        scan_array = np.clip(scan_array, 0.0, self.max_lidar_range)

        # Binning
        points_per_bin = len(scan_array) // self.n_lidar_bins
        binned_scan = []

        for i in range(self.n_lidar_bins):
            start_idx = i * points_per_bin
            end_idx = (i + 1) * points_per_bin if i < self.n_lidar_bins - 1 else len(scan_array)
            bin_min = np.min(scan_array[start_idx:end_idx])
            binned_scan.append(bin_min)

        # Normalizar a [0,1]
        return np.array(binned_scan, dtype=np.float32) / self.max_lidar_range

    def compute_goal_info(self, current_pos, goal_pos, current_yaw):
        """
        Distancia y ángulo relativo al goal, normalizados.
        current_pos: (x, y)
        goal_pos: (x, y)
        current_yaw: orientación actual (rad)
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]

        distance = float(np.sqrt(dx**2 + dy**2))

        angle_to_goal = np.arctan2(dy, dx)
        relative_angle = angle_to_goal - current_yaw

        # Normalizar ángulo a [-pi, pi]
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

        distance_norm = np.clip(distance / 10.0, 0.0, 1.0)  # asumir 10m máx
        angle_norm = relative_angle / np.pi  # [-1,1]

        return np.array([distance_norm, angle_norm], dtype=np.float32)

    def get_state(self, scan_data, current_pos, goal_pos, current_yaw):
        """Devuelve estado completo: [bins LiDAR, dist_goal_norm, ang_goal_norm]."""
        lidar_state = self.process_lidar(scan_data)
        goal_state = self.compute_goal_info(current_pos, goal_pos, current_yaw)
        return np.concatenate([lidar_state, goal_state], axis=0)
