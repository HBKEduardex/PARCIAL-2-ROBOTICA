# test_node.py
import rclpy
from rclpy.node import Node
import numpy as np

from .environment import TurtleBot3Env
from .state_processor import StateProcessor
from .dqn_agent import DQNAgent


class TestNode(Node):
    """Evalúa un modelo entrenado DQN en TurtleBot3 con filtros suaves en test."""

    def __init__(self):
        super().__init__("dqn_test_node")

        # Dimensión del estado: 10 bins LiDAR + 2 del goal = 12
        self.n_lidar_bins = 10
        self.state_size = self.n_lidar_bins + 2
        self.action_size = 5

        # Cargar modelo entrenado
        self.agent = DQNAgent(
            state_dim=self.state_size,
            n_actions=self.action_size,
        )
        # 👇 Ajusta la ruta si quieres probar otro modelo
        self.agent.load(
            "/home/eduardex/Documents/Robotica/parcial2_ws/trained_model.pkl"
        )

        # En test el agente NO explora (siempre política greedy)
        self.agent.epsilon = 0.0

        # Entorno y procesador de estado
        self.env = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=self.n_lidar_bins)

        # Configuración de test
        self.num_test_episodes = 10
        self.max_steps = 1000

        # Para filtro anti-zigzag
        self.last_action = None

    def get_state(self):
        """Devuelve el estado procesado."""
        scan = self.env.scan_data
        if scan is None:
            # Si aún no hay LiDAR, asumimos todo libre (max range)
            scan = [self.state_processor.max_lidar_range] * 360

        return self.state_processor.get_state(
            scan_data=scan,
            current_pos=self.env.position,
            goal_pos=self.env.goal_position,
            current_yaw=self.env.yaw,
        )

    def filtered_action(self, raw_action: int) -> int:
        """
        Filtro suave anti-zigzag:
        - Si el agente quiere alternar muy rápido entre girar izq (1) y der (2),
          podemos mantener la acción anterior en vez de cambiar cada paso.
        """
        action = raw_action

        if (
            self.last_action is not None
            and raw_action in [1, 2]
            and self.last_action in [1, 2]
            and raw_action != self.last_action
        ):
            # Aquí estaría el “baile L-R-L-R”.
            # Estrategia: en vez de cambiar inmediatamente de lado,
            # repetimos la última acción una vez.
            self.get_logger().debug(
                f"🔁 Anti-zigzag: raw_action={raw_action}, "
                f"last_action={self.last_action} -> usando last_action"
            )
            action = self.last_action

        return action

    def test(self):
        self.get_logger().info("==== 🚀 INICIANDO EVALUACIÓN DEL MODELO ====")

        successes = 0
        all_final_distances = []
        all_rewards = []

        for ep in range(self.num_test_episodes):
            self.get_logger().info(
                f"\n===== EPISODIO DE TEST {ep+1}/{self.num_test_episodes} ====="
            )

            # Reset aleatorio del entorno
            self.env.reset(random_goal=True)
            rclpy.spin_once(self.env, timeout_sec=0.5)

            # Objetivo de este episodio
            goal_x, goal_y = self.env.goal_position
            self.get_logger().info(
                f"🎯 Objetivo del episodio {ep+1}: x = {goal_x:.2f}, y = {goal_y:.2f}"
            )

            # Posición inicial del robot
            start_x, start_y = self.env.position
            self.get_logger().info(
                f"📍 Posición inicial robot: x = {start_x:.2f}, y = {start_y:.2f}"
            )

            state = self.get_state()
            total_reward = 0.0
            self.last_action = None  # reiniciamos historial por episodio

            for step in range(self.max_steps):
                # Acción propuesta por el DQN (política greedy)
                raw_action = self.agent.act(state)

                # Aplicamos filtro anti-zigzag en test
                action = self.filtered_action(raw_action)

                reward, done = self.env.step(action)
                total_reward += reward

                next_state = self.get_state()
                state = next_state

                # Guardamos la acción que realmente se ejecutó
                self.last_action = action

                if done:
                    break

            # Posición final del robot
            end_x, end_y = self.env.position
            final_dist = self.env.distance_to_goal()
            all_final_distances.append(final_dist)
            all_rewards.append(total_reward)

            self.get_logger().info(
                f"📍 Posición final robot: x = {end_x:.2f}, y = {end_y:.2f}"
            )
            self.get_logger().info(
                f"📏 Distancia final al objetivo: {final_dist:.2f} m"
            )

            # Checamos cómo terminó el episodio
            if self.env.is_goal_reached():
                self.get_logger().info(f"🎯 Episodio {ep+1}: OBJETIVO ALCANZADO")
                successes += 1
            else:
                # Puede ser colisión o simplemente timeout
                if self.env.is_collision():
                    reason = "COLISIÓN"
                else:
                    reason = "SIN ÉXITO (tiempo agotado)"

                self.get_logger().info(f"💥 Episodio {ep+1}: {reason}")

            self.get_logger().info(
                f"🏁 Reward total del episodio: {total_reward:.2f}"
            )

        success_rate = (successes / self.num_test_episodes) * 100.0
        avg_dist = float(np.mean(all_final_distances)) if all_final_distances else 0.0
        avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        self.get_logger().info("\n==== RESULTADO FINAL ==== ")
        self.get_logger().info(f"✔ Éxitos: {successes}/{self.num_test_episodes}")
        self.get_logger().info(f"✔ Tasa de éxito: {success_rate:.2f}%")
        self.get_logger().info(f"📏 Distancia final promedio al goal: {avg_dist:.2f} m")
        self.get_logger().info(f"⭐ Reward promedio por episodio: {avg_reward:.2f}")
        self.get_logger().info("Evaluación terminada.")


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        node.test()
    finally:
        node.env.send_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
