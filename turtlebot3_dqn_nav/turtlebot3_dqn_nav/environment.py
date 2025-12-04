# environment.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import numpy as np
import math

class TurtleBot3Env(Node):
    """Entorno ROS2 para navegación autónoma del TurtleBot3 usando DQN."""

    def __init__(self):
        super().__init__("turtlebot3_env")

        # ---------------------------
        #  Publishers / Subscribers
        # ---------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        # Servicio para resetear Gazebo
        self.reset_world_client = self.create_client(Empty, "/reset_world")

        # ---------------------------
        #  Estado interno
        # ---------------------------
        self.scan_data = None
        self.position = (0.0, 0.0)
        self.yaw = 0.0

        # Objetivo inicial (se redefinirá en reset)
        self.goal_position = (0.0, 0.0)

        # Acciones discretas
        # (líneal, angular)
        self.actions = {
            0: (0.15, 0.00),   # adelante recto
            1: (0.00, 0.15),   # giro suave izquierda
            2: (0.00, -0.15),  # giro suave derecha
            3: (0.10, 0.10),   # arco izquierdo
            4: (0.10, -0.10),  # arco derecho
        }

        # ---------------------------
        #  Parámetros de entorno
        # ---------------------------
        self.collision_threshold = 0.20    # distancia mínima a obstáculos
        self.goal_threshold = 0.9       # radio de éxito (más realista)
        self.last_distance = None
        self.last_action = None

    # ============================================================
    #  CALLBACKS
    # ============================================================

    def scan_callback(self, msg: LaserScan):
        self.scan_data = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

        # Convertir quaternion → yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    # ============================================================
    #  FUNCIONES DE APOYO
    # ============================================================

    def send_velocity(self, linear: float, angular: float):
        """Publicar velocidad al robot."""
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(twist)

    def distance_to_goal(self) -> float:
        """Distancia euclidiana al objetivo."""
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        return float(math.sqrt(dx * dx + dy * dy))

    def is_collision(self) -> bool:
        """Detecta si la distancia mínima del LiDAR indica colisión."""
        if self.scan_data is None:
            return False

        arr = np.array(self.scan_data, dtype=np.float32)
        arr[np.isinf(arr)] = 999.0
        arr[np.isnan(arr)] = 999.0

        return float(np.min(arr)) < self.collision_threshold

    def is_goal_reached(self) -> bool:
        """Verifica si el robot está dentro del radio de éxito."""
        return self.distance_to_goal() < self.goal_threshold

    # ============================================================
    #  REWARD
    # ============================================================

    def compute_reward(self, action_idx: int) -> float:
        """Función de recompensa con progreso + penalizaciones suaves."""

        current_dist = self.distance_to_goal()

        # Progreso hacia el objetivo
        if self.last_distance is None:
            progress_reward = 0.0
        else:
            progress = self.last_distance - current_dist
            progress_reward = progress * 30.0  # refuerzo fuerte

        self.last_distance = current_dist

        # Penalización por obstáculos cercanos
        if self.scan_data is not None:
            arr = np.array(self.scan_data, dtype=np.float32)
            arr[np.isinf(arr)] = 3.5
            arr[np.isnan(arr)] = 3.5
            min_d = float(np.min(arr))
        else:
            min_d = 3.5

        if min_d < 0.5:
            obstacle_penalty = -5.0 * (0.5 - min_d)
        else:
            obstacle_penalty = 0.0

        # Penalización por giro constante sin avanzar
        action_penalty = -0.01 if action_idx in [1, 2] else 0.0

        # Penalización temporal para evitar loops infinitos
        time_penalty = -0.02

        return float(progress_reward + obstacle_penalty + action_penalty + time_penalty)

    # ============================================================
    #  RESET
    # ============================================================

    def reset_world(self):
        """Llamar al servicio de Gazebo /reset_world."""
        if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Servicio /reset_world no disponible")
            return

        req = Empty.Request()
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.result() is None:
            self.get_logger().error("❌ Fallo al resetear mundo")
        else:
            self.get_logger().info("🌍 Mundo reseteado correctamente")

    def reset(self, random_goal: bool = True):
        """Resetea el entorno del episodio."""

        # Detener robot
        self.send_velocity(0.0, 0.0)

        # Resetear Gazebo
        self.reset_world()

        # ---------- Generar nuevo goal dentro de un área segura -----------
        if random_goal:
            x_min, x_max = -1.5, 1.5
            y_min, y_max = -1.5, 1.5

            gx = float(np.random.uniform(x_min, x_max))
            gy = float(np.random.uniform(y_min, y_max))

            self.goal_position = (gx, gy)

            self.get_logger().info(
                f"🎯 Nuevo objetivo aleatorio: x={gx:.2f}, y={gy:.2f}"
            )

        # Esperar que lleguen LiDAR y odometría actualizados
        self.scan_data = None
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.scan_data is not None:
                break

        self.last_distance = self.distance_to_goal()
        self.last_action = None

    # ============================================================
    #  STEP
    # ============================================================

    def step(self, action_idx: int):
        """
        Ejecuta una acción discreta y devuelve:
        - reward
        - done
        """

        linear, angular = self.actions[action_idx]
        self.send_velocity(linear, angular)

        # Dejar que el robot avance y llegue un nuevo estado
        rclpy.spin_once(self, timeout_sec=0.1)

        # ---------- ORDEN CORRECTO: primero goal, luego colisión ----------
        if self.is_goal_reached():
            reward = 200.0
            done = True
            self.get_logger().info("🎯 Goal alcanzado")
        elif self.is_collision():
            reward = -100.0
            done = True
            self.get_logger().info("💥 Colisión")
        else:
            reward = self.compute_reward(action_idx)
            done = False

        self.last_action = action_idx
        return float(reward), bool(done)
