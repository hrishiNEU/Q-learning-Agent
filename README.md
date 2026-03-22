# Deep Q-Learning Agent for Atari Breakout

A Deep Q-Learning (DQN) agent trained to play Atari Breakout using PyTorch and OpenAI Gymnasium. Built as part of a reinforcement learning assignment exploring how agents learn through environment interaction.

**Author:** Hrishikesh Kulkarni

---

## Demo

> Agent gameplay video and training metrics chart available in the repository.

---

## Results

| Metric | Value |
|---|---|
| Episodes trained | 100 |
| Average reward | 1.45 |
| Max reward | 7.00 |
| Average steps/episode | 202.0 |
| Final epsilon | 0.1413 |

---

## How It Works

The agent learns to play Breakout entirely through trial and error — it receives a reward when it breaks a brick and receives no reward otherwise. Over time, it learns which paddle movements lead to more bricks destroyed.

Key techniques used:
- **Convolutional Neural Network** to process raw game frames
- **Experience Replay** to learn from past transitions
- **Target Network** for stable Bellman updates
- **ε-greedy exploration** with exponential decay
- **Reward clipping** to [-1, 1] for training stability

---

## Project Structure

```
├── dqn_breakout.ipynb       # Main Colab notebook (all code)
├── dqn_breakout.pth         # Trained model weights
├── dqn_breakout_metrics.png # Training metrics chart
├── agent_play.mp4           # Recorded gameplay video
└── README.md
```

---

## Setup & Usage

### Run in Google Colab (Recommended)
1. Open `dqn_breakout.ipynb` in Google Colab
2. Set runtime to **T4 GPU** via Runtime → Change runtime type
3. Run all cells in order (Runtime → Run all)

### Run Locally
```bash
pip install gymnasium[atari] ale-py torch matplotlib
pip install "gymnasium[accept-rom-license]"
pip install pyvirtualdisplay imageio[ffmpeg]
```
Then run the notebook cell by cell.

---

## Architecture

```
Input: 4 stacked grayscale frames (4 × 84 × 84)
         ↓
Conv2d(4→32, kernel=8, stride=4)  + ReLU
Conv2d(32→64, kernel=4, stride=2) + ReLU
Conv2d(64→64, kernel=3, stride=1) + ReLU
         ↓
Flatten → Linear(3136→512) + ReLU → Linear(512→4)
         ↓
Output: Q-value for each of 4 actions
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 0.0001 |
| Gamma (γ) | 0.99 |
| Epsilon start | 1.0 |
| Epsilon min | 0.01 |
| Decay rate | 0.0001 |
| Batch size | 32 |
| Replay buffer | 10,000 |
| Target sync | every 1,000 steps |

---

## License

This project is released under the **MIT License** — see the license header in `dqn_breakout.ipynb` for full details.

**Third-party licenses:**
- OpenAI Gymnasium — MIT License
- Arcade Learning Environment — GPL-2.0
- PyTorch — BSD-style License

---

## References

- Mnih et al. (2015), *Human-level control through deep reinforcement learning*, Nature
- OpenAI Baselines — https://github.com/openai/baselines
- PyTorch RL Tutorial — https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

