# Multi-Objective PPO-Based Receiver Energy- and SIC-Complexity-Aware NOMA Scheduling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29.1-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.2.1-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Machine Learning Course Project (25SC2107E) — Department of Electronics & Communication Engineering, KL University**  
> **Authors:** Ranga Charvik Battagiri, A. Dhanush, R. Gnaneshwar  
> **Faculty Supervisor:** Mr. N Lava Prasad, Assistant Professor, Department of ECE  

---

## Table of Contents
1. [Executive Summary & Motivation](#-executive-summary--motivation)
2. [Key Contributions](#-key-contributions)
3. [System Architecture & Mathematical Formulation](#-system-architecture--mathematical-formulation)
   - [Downlink NOMA Superposition & Power Allocation](#1-downlink-noma-superposition--power-allocation)
   - [Channel Fading, Imperfect CSI & Multi-Cell Interference](#2-channel-fading-imperfect-csi--multi-cell-interference)
   - [SINR & Shannon Achievable Rate](#3-sinr--shannon-achievable-rate)
   - [Hardware-Aware Receiver Energy & Urgency Model](#4-hardware-aware-receiver-energy--urgency-model)
   - [Fairness Metric](#5-fairness-metric)
4. [Reinforcement Learning Formulation (MDP)](#-reinforcement-learning-formulation-mdp)
   - [State Space](#state-space)
   - [Action Space](#action-space)
   - [Multi-Objective Composite Reward Function](#multi-objective-composite-reward-function)
   - [Agent Architecture & Hyperparameters](#agent-architecture--hyperparameters)
5. [Repository Structure](#-repository-structure)
6. [Dataset & Exploratory Data Analysis (EDA)](#-dataset--exploratory-data-analysis-eda)
7. [Benchmark Evaluation & Comparative Results](#-benchmark-evaluation--comparative-results)
8. [Installation & Execution Guide](#-installation--execution-guide)
9. [References](#-references)
10. [License](#-license)

---

## 📌 Executive Summary & Motivation

The exponential transition towards 5G-Advanced and 6G networks has catalyzed massive Machine-Type Communication (mMTC), interconnecting billions of resource-constrained Internet of Things (IoT) sensors, edge processors, and wearable devices. Traditional **Orthogonal Multiple Access (OMA)** schemes—such as TDMA and OFDMA—strictly allocate mutually exclusive time slots or frequency subcarriers to single users. Although OMA eliminates inter-user interference, it introduces a hard ceiling on concurrent connections bounded by the number of orthogonal radio resource blocks.

**Power-Domain Non-Orthogonal Multiple Access (NOMA)** overcomes this limit by allowing multiple users to simultaneously share the identical frequency-time block via power-domain multiplexing. At the base station, signals are superimposed with distinct power allocations. At the receiving end, **Successive Interference Cancellation (SIC)** is executed: users iteratively demodulate, decode, and subtract the strongest interfering signals before extracting their own information.

### The Core Problem: The Hidden Energy Cost of SIC
While increasing NOMA cluster sizes boosts aggregate spectral efficiency, it imposes an extreme computational and thermal toll on user equipment. For a cluster of size $C$, strong receivers must perform up to $C - 1$ decoding and subtraction stages[cite: 45, 56]. In real-world IoT environments, existing network schedulers are **"battery-blind"**—they optimize strictly for aggregate Shannon throughput or static fairness. Consequently, an IoT node with critical battery depletion (e.g., $<5\%$) is frequently assigned to large, dense NOMA clusters simply because its instantaneous channel conditions favor spectral efficiency, inducing rapid battery death and premature network node dropouts[cite: 45].

This project designs, trains, and evaluates a **Multi-Objective Proximal Policy Optimization (PPO)** Reinforcement Learning scheduler that dynamically observes real-time user battery levels alongside instantaneous channel gains to build energy-aware, complexity-constrained NOMA clusters[cite: 45, 56].

---

## 💡 Key Contributions

* **State-Space Battery Visibility:** Formulates an MDP observation space incorporating both real-time channel gains and residual battery percentages ($b_i \in [0, 1]$) across all active users[cite: 45, 56].
* **Dynamic Energy Urgency Penalty:** Implements a steep battery-urgency penalty multiplier $(1 + 3 \cdot [1 - b_i])$ that scales receiver energy overhead up to $4\times$ when devices approach critical discharge, teaching the agent to dynamically isolate low-battery nodes into low-complexity clusters or OMA-like allocations[cite: 45, 56].
* **Realistic Physical Layer Simulation:** Integrates Gauss-Markov channel fading, imperfect Channel State Information (CSI) estimation noise, and stochastic multi-cell interference within a custom Gymnasium simulation environment (`noma_env.py`)[cite: 45, 56].
* **Benchmarked Superiority:** Evaluated across 30 independent test episodes against standard Random, Fixed-slice, and Channel-Gain Greedy baseline heuristics, demonstrating an overall **51% performance gain** in combined multi-objective utility.

---

## ⚙️ System Architecture & Mathematical Formulation

The modelled network topology comprises a single Base Station (BS) serving $N = 6$ downlink users categorized dynamically across $K = 3$ orthogonal sub-band clusters[cite: 45, 56].

### 1. Downlink NOMA Superposition & Power Allocation
Users mapped to cluster $k$ are designated by index subset $\mathcal{U}_k \subseteq \{1, \dots, N\}$[cite: 56]. The base station transmits a superimposed composite signal $x_k$ over cluster $k$:

$$x_k = \sum_{i \in \mathcal{U}_k} \sqrt{p_i} \, s_i$$

where $s_i$ denotes the information symbol for user $i$ and $p_i$ is its allocated transmit power such that $\sum_{i \in \mathcal{U}_k} p_i = P_{\text{total}}$[cite: 45, 56]. 

Transmit power shares are allocated inversely proportional to perceived channel gains ($\tilde{h}_i$), ensuring weaker channel devices receive adequate power to satisfy Quality-of-Service constraints[cite: 45, 56]:

$$p_i = \frac{\tilde{h}_i^{-1}}{\sum_{j \in \mathcal{U}_k} \tilde{h}_j^{-1}} P_{\text{total}}$$

### 2. Channel Fading, Imperfect CSI & Multi-Cell Interference
To replicate true mobile conditions, physical channels experience Gauss-Markov temporal fading[cite: 56]:

$$h_i^{(t+1)} = \text{clip}\left(h_i^{(t)} + \mathcal{N}(0, \sigma_{\text{fade}}^2),\, 0.05,\, 1.0\right), \quad \sigma_{\text{fade}} = 0.05$$

The base station allocates power based on estimated (perceived) gains affected by estimation error $\epsilon_i \sim \mathcal{N}(0, \sigma_{\text{csi}}^2)$ with $\sigma_{\text{csi}} = 0.02$[cite: 56]:

$$\tilde{h}_i = \text{clip}\left(h_i + \epsilon_i,\, 0.01,\, 1.0\right)$$

Additionally, out-of-cell interference from surrounding cellular sectors is injected dynamically as $I_{\text{multi}} \sim \mathcal{U}(0.0, 0.05)$[cite: 56].

### 3. SINR & Shannon Achievable Rate
For user $i$ inside cluster $k$, assuming successive cancellation of signals weaker than its own, the instantaneous Signal-to-Interference-plus-Noise Ratio (SINR) is given by[cite: 45, 56]:

$$\text{SINR}_i = \frac{p_i h_i}{\left(\sum_{j \in \mathcal{U}_k, \, j \neq i} p_j\right) h_i + \sigma_{\text{noise}}^2 + I_{\text{multi}}}$$

The achievable data rate (throughput) across channel bandwidth $B$ is determined via Shannon capacity[cite: 45, 56]:

$$R_i = B \log_2\left(1 + \text{SINR}_i\right)$$

Total network throughput is defined as $R_{\text{total}} = \sum_{i=1}^N R_i$[cite: 45, 56].

### 4. Hardware-Aware Receiver Energy & Urgency Model
Each receiver consumes a baseline baseband wake-up energy $E_{\text{base}} = 0.01$ units[cite: 45, 56]. Each successive cancellation decoding iteration adds an energy surcharge of $E_{\text{stage}} = 0.05$ units[cite: 45, 56]. For a cluster of size $C_k = \vert{}\mathcal{U}_k\vert{}$, the required SIC decoding stages are[cite: 45, 56]:

$$\text{SIC}_i = \max(C_k - 1, 0)$$

The unweighted raw energy expenditure is:

$$E_{\text{raw}, i} = E_{\text{base}} + \left(\text{SIC}_i \cdot E_{\text{stage}}\right)$$

To penalize assigning energy-depleted nodes to multi-user clusters, we define battery urgency $u_i = 1.0 - b_i$, where $b_i \in [0, 1]$ represents remaining battery[cite: 45, 56]. The weighted receiver cost evaluated by the scheduler is[cite: 45, 56]:

$$E_i = E_{\text{raw}, i} \times (1.0 + 3.0 \cdot u_i)$$

At $b_i \to 0$ ($u_i \to 1$), the energy cost receives a **$4\times$ penalty multiplication**[cite: 45]. User battery states deplete per transmission step according to $b_i^{(t+1)} = \max\left(0.0,\, b_i^{(t)} - 0.1 \cdot E_{\text{raw}, i}\right)$[cite: 56].

### 5. Fairness Metric
Network resource distribution is tracked using **Jain's Fairness Index** over user rates $R_i$[cite: 45, 56]:

$$\mathcal{J} = \frac{\left(\sum_{i=1}^N R_i\right)^2}{N \sum_{i=1}^N R_i^2 + 10^{-9}}$$

---

## 🤖 Reinforcement Learning Formulation (MDP)

The user-to-cluster scheduling task is formulated as a discrete-time Markov Decision Process $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$[cite: 45]:
