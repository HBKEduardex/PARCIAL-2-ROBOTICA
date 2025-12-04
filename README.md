\# ===============================================================

\# README – Navegación Autónoma con DQN – TurtleBot3

\# Autores:

\# - Adrián Eduardo Vargas Llanquipacha

\# - Israel Silva Bernal

\# ===============================================================

Este proyecto implementa un agente Deep Q-Network (DQN) que permite al robot

TurtleBot3 Burger navegar en un entorno simulado en Gazebo, acercándose a

objetivos aleatorios y evitando obstáculos usando únicamente un sensor LiDAR 2D.

\# ===============================================================

\# 🚀 EJECUCIÓN DEL PROYECTO

\# ===============================================================

\# ---------------------------------------------------------------

\# 1️⃣ COMPILAR EL PAQUETE

\# ---------------------------------------------------------------

cd ~/parcial2\_ws

colcon build --packages-select turtlebot3\_dqn\_nav

source install/setup.bash

\# ---------------------------------------------------------------

\# 2️⃣ INICIAR EL MUNDO EN GAZEBO

\# ---------------------------------------------------------------

ros2 launch turtlebot3\_gazebo turtlebot3\_world.launch.py

\# ---------------------------------------------------------------

\# 3️⃣ ENTRENAR EL MODELO

\# ---------------------------------------------------------------

cd ~/parcial2\_ws

source install/setup.bash

ros2 run turtlebot3\_dqn\_nav train\_node

\# El modelo entrenado se guarda automáticamente como:

trained\_model.pkl

\# ---------------------------------------------------------------

\# 4️⃣ EJECUTAR LA EVALUACIÓN DEL AGENTE ENTRENADO

\# ---------------------------------------------------------------

cd ~/parcial2\_ws

source install/setup.bash

ros2 run turtlebot3\_dqn\_nav test\_node

\# Durante la evaluación se muestra:

\# 🎯 Objetivo generado

\# 📍 Posición inicial del robot

\# 📍 Posición final del robot

\# 📏 Distancia final al objetivo

\# 💥 Resultado (ÉXITO / COLISIÓN / TIMEOUT)

\# 📊 Estadísticas finales de 10 episodios

\# ===============================================================

\# ✔️ ESTADO ACTUAL DEL MODELO (trained\_model.pkl)

\# ===============================================================

\# El modelo incluido alcanza:

\# - 50% de éxito (radio de aceptación 0.9 m)

\# - Movimientos estables sin zig-zag excesivo

\# - Reducción significativa de colisiones

\# - Aproximación consistente al objetivo

\# ===============================================================

\# 📝 NOTAS FINALES

\# ===============================================================

\# ✔ Se cumple el requerimiento mínimo del 30% de éxito.

\# ✔ El robot utiliza únicamente el LiDAR 2D como entrada sensorial.

\# ✔ No se emplean mapas ni algoritmos de path planning clásicos.

\# ✔ El aprendizaje se basa exclusivamente en interacción con el entorno.

\# ✔ El sistema funciona dentro del mundo turtlebot3\_world de Gazebo.

\# ===============================================================

\# FIN DEL README

\# ===============================================================
