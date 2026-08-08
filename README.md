# 🎮 Real-Time Multi-Agent Strategic Navigation using Deep Reinforcement Learning

M.Tech Minor Project | Deep Q-Network (DQN) | Pygame | Multi-Agent Systems | Reinforcement Learning

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green)

---

## 📌 Overview

This project implements a **real-time multi-agent maze environment** where intelligent agents learn strategic navigation and decision-making using **Deep Q-Networks (DQN)**. The system combines reinforcement learning with visual deception, dynamic maze generation, power-ups, enemies, traps, and treasure collection.

The project was developed as part of my **M.Tech (Artificial Intelligence) minor project** and is associated with a **published research paper**.

---

## ✨ Features

- 🧠 **Deep Q-Network (DQN)** using PyTorch
- 👥 **Multi-agent environment**
- 🌲 **Procedural maze generation**
- 👻 Stealth, ghost, invincibility, boost, and disarm power-ups
- 💎 Gems, treasures, fake treasures, enemies, and spike traps
- 📊 Reward shaping and performance analytics
- 🎮 Real-time rendering with Pygame
- 🔄 Dynamic maze reshaping during gameplay
- 💾 Save/load player profiles

---

## 🧠 Reinforcement Learning Components

| Component | Description |
|---|---|
| State Space | Agent position, environment entities, power-up states |
| Action Space | Move, stealth, ghost, invincibility, boost, disarm |
| Algorithm | Deep Q-Network (DQN) |
| Replay Buffer | Experience replay |
| Target Network | Stable Q-learning updates |
| Reward Function | Treasure rewards, penalties, exploration shaping |

---

## 🛠️ Technologies Used

- **Python 3**
- **PyTorch**
- **Pygame**
- **NumPy**
- **JSON**
- **Deque / Replay Buffer**

---

## 📂 Project Structure

```text
.
├── 911finalminiproject.py
├── README.md
├── LICENSE
├── assets/              # Optional: sounds, images, sprites
├── models/              # Optional: trained DQN weights
└── paper/               # Optional: published paper PDF
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/multi-agent-dqn-maze-game.git
cd multi-agent-dqn-maze-game
```

### 2. Install dependencies

```bash
pip install pygame numpy torch
```

### 3. Run the project

```bash
python 911finalminiproject.py
```

---

## 🎮 Gameplay

- Collect treasures and gems
- Avoid enemies and spike traps
- Use power-ups strategically
- Reach the main treasure to win
- Agents learn through reinforcement learning

---

## 📊 Research Contribution

This work explores:

- Multi-agent reinforcement learning
- Strategic navigation
- Visual deception
- Dynamic environments
- Reward shaping for complex tasks

---

## 📄 Published Paper

**Title:** *Add your exact published paper title here*

**Conference / Journal:** *Add conference name*

**Year:** 2026

If available, add the paper PDF in the `paper/` folder and link it here.

---

## 📸 Screenshots

### Main Game Screen

_Add screenshot_

### DQN Training / Analytics

_Add screenshot_

### Maze and Power-Ups

_Add screenshot_

---

## 👩‍💻 Author

**S. Safrin Fathima**

- M.Tech in Artificial Intelligence
- CVR College of Engineering
- Hyderabad, India

GitHub: https://github.com/your-username

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgements

- PyTorch
- Pygame
- Open-source reinforcement learning community
