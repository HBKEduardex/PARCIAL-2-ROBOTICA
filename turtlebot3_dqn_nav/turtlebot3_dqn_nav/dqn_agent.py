# dqn_agent.py
import numpy as np
import random
from collections import deque
from copy import deepcopy
from sklearn.neural_network import MLPRegressor
import joblib


class DQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,      # se deja por compatibilidad, ya no se usa dentro
        buffer_size=50000,
        batch_size=64,
        target_update_freq=1000,
        seed=0
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr

        # ============================
        # Manejo de epsilon (exploración)
        # ============================
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start

        # Número de episodios en los que queremos ir
        # de epsilon_start a epsilon_end (decay lineal)
        self.epsilon_decay_episodes = 400

        # Se deja guardado por si en algún momento lo usas,
        # pero dentro de esta clase ya no se aplica por step.
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0

        random.seed(seed)
        np.random.seed(seed)
        self.memory = deque(maxlen=buffer_size)

        self.q_net = self._build_model()
        self.target_net = self._build_model()
        self._warmup_networks()
        self._update_target_network()

    def _build_model(self):
        model = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            solver="adam",
            learning_rate_init=self.lr,
            max_iter=1,
            warm_start=True,
            random_state=0
        )
        return model

    def _warmup_networks(self):
        X_dummy = np.zeros((1, self.state_dim))
        y_dummy = np.zeros((1, self.n_actions))
        self.q_net.fit(X_dummy, y_dummy)
        self.target_net.fit(X_dummy, y_dummy)

    def _update_target_network(self):
        self.target_net = deepcopy(self.q_net)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Política epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        state = np.array(state, dtype=np.float32).reshape(1, -1)
        q_values = self.q_net.predict(state)[0]
        return int(np.argmax(q_values))

    def _sample_minibatch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)

        return states, actions, rewards, next_states, dones

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self._sample_minibatch()

        # Q(s,a) actual
        q_values = self.q_net.predict(states)

        # Q_target(s', a') usando la red target
        q_next = self.target_net.predict(next_states)
        max_q_next = np.max(q_next, axis=1)

        # Construcción de targets
        targets = q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q = rewards[i]
            else:
                target_q = rewards[i] + self.gamma * max_q_next[i]
            targets[i, actions[i]] = target_q

        # Actualizar la red principal
        self.q_net.partial_fit(states, targets)

        # ⚠️ IMPORTANTE: ya NO actualizamos epsilon aquí.
        # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self._update_target_network()

    def update_epsilon(self, episode: int):
        """
        Actualiza epsilon EN FUNCIÓN DEL EPISODIO (no de cada step).

        Decaimiento lineal:
        - episode = 0  → epsilon ≈ epsilon_start
        - episode = epsilon_decay_episodes → epsilon ≈ epsilon_end
        - después se queda en epsilon_end
        """
        frac = min(1.0, episode / float(self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)

    def save(self, path="trained_model.pkl"):
        joblib.dump(
            {
                "q_net": self.q_net,
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_episodes": self.epsilon_decay_episodes,
            },
            path
        )

    def load(self, path="trained_model.pkl"):
        data = joblib.load(path)
        self.q_net = data["q_net"]
        self._update_target_network()

        # Recuperar estado si existe
        self.epsilon = data.get("epsilon", self.epsilon)
        self.train_steps = data.get("train_steps", 0)
        self.epsilon_start = data.get("epsilon_start", self.epsilon_start)
        self.epsilon_end = data.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay_episodes = data.get("epsilon_decay_episodes", self.epsilon_decay_episodes)
