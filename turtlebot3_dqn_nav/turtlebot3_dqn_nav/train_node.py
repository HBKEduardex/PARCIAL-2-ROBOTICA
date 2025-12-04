# train_node.py
import rclpy
from rclpy.node import Node
import numpy as np
from .environment import TurtleBot3Env
from .state_processor import StateProcessor
from .dqn_agent import DQNAgent


class TrainNode(Node):
    """Nodo de entrenamiento DQN parecido a tu guía de laboratorio."""

    def __init__(self):
        super().__init__('dqn_training_node')

        # Parámetros de entrenamiento (puedes ajustar)
        self.n_episodes = 500          # mínimo 200 como en el enunciado
        self.max_steps_per_episode = 300

        # Estado: 10 bins LiDAR + 2 de goal
        self.n_lidar_bins = 10
        self.state_size = self.n_lidar_bins + 2
        self.action_size = 5

        # Componentes
        self.env = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=self.n_lidar_bins)

        self.agent = DQNAgent(
            state_dim=self.state_size,
            n_actions=self.action_size,
            gamma=0.99,
            lr=1e-3,
            epsilon_start=1.0,
            epsilon_end=0.01,      # se usará en el schedule por episodio
            epsilon_decay=0.995,   # ya no se usa por step, se queda por compat
            buffer_size=10000,
            batch_size=64,
            target_update_freq=1000,
        )

        # Logs de stats
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0

    def get_processed_state(self):
        """Combina LiDAR + odom + goal usando StateProcessor."""
        if self.env.scan_data is None:
            # si por alguna razón no hay scan, devolver algo neutro
            scan = [self.state_processor.max_lidar_range] * 360
        else:
            scan = self.env.scan_data

        return self.state_processor.get_state(
            scan_data=scan,
            current_pos=self.env.position,
            goal_pos=self.env.goal_position,
            current_yaw=self.env.yaw,
        )

    def train(self):
        self.get_logger().info("==== 🚀 Iniciando entrenamiento DQN (modo guía) ====")

        for ep in range(self.n_episodes):
            self.get_logger().info(f"\n===== EPISODIO {ep+1}/{self.n_episodes} =====")

            # Reset entorno + goal aleatorio
            self.env.reset(random_goal=True)
            rclpy.spin_once(self.env, timeout_sec=0.5)

            state = self.get_processed_state()
            episode_reward = 0.0

            for step in range(self.max_steps_per_episode):
                # Seleccionar acción (ε-greedy)
                action = self.agent.act(state)

                # Ejecutar acción
                reward, done = self.env.step(action)

                # Obtener siguiente estado
                next_state = self.get_processed_state()

                # Guardar transición y entrenar
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.train_step()

                episode_reward += reward
                state = next_state

                if done:
                    break

            # 🔥 actualizar epsilon EN FUNCIÓN DEL EPISODIO (no de cada step)
            self.agent.update_epsilon(ep)

            # Estadísticas de episodio
            self.episode_rewards.append(episode_reward)

            if self.env.is_goal_reached():
                self.success_count += 1
            elif self.env.is_collision():
                self.collision_count += 1

            avg_last_10 = (
                np.mean(self.episode_rewards[-10:])
                if len(self.episode_rewards) >= 10
                else episode_reward
            )
            success_rate = (self.success_count / (ep + 1)) * 100.0

            self.get_logger().info(
                f"🏁 Episodio {ep+1}: Reward={episode_reward:.2f} | "
                f"Avg(10)={avg_last_10:.2f} | "
                f"ε={self.agent.epsilon:.3f} | "
                f"SuccessRate={success_rate:.1f}%"
            )

        # Guardar modelo final
        self.agent.save('trained_model.pkl')
        self.get_logger().info("💾 Modelo guardado en trained_model.pkl")


def main(args=None):
    rclpy.init(args=args)
    node = TrainNode()
    try:
        node.train()
    finally:
        node.env.send_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
