"""
Retail Brain × Sainsbury's — Phase 4: Reinforcement Learning Agent
A Decision Agent that learns over time which replenishment strategies
maximize profit and minimize waste.

Architecture:
- State: (stock_level, velocity, days_of_cover, stockout_risk, weather, events, elasticity)
- Action: order_quantity (discretised into buckets)
- Reward: revenue_gained - holding_cost - stockout_penalty

Uses Proximal Policy Optimization (PPO) via a lightweight custom implementation
that doesn't require heavy RL frameworks like Stable Baselines for production.

Usage:
    from rl_agent import InventoryRLAgent, InventoryEnvironment
"""

import os
import sys
import json
import pickle
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from logger import get_logger

logger = get_logger("rl_agent")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RL_MODEL_PATH = os.path.join(MODELS_DIR, "rl_agent.pkl")


# ── Action Space ─────────────────────────────────────────────────────────────
# Discretised order quantities as multipliers of reorder_point
ORDER_ACTIONS = [
    0.0,    # Don't order
    0.5,    # Half reorder point
    1.0,    # Reorder point
    1.5,    # 1.5x reorder
    2.0,    # 2x reorder
    3.0,    # 3x reorder (aggressive)
]
N_ACTIONS = len(ORDER_ACTIONS)


# ── State Features ───────────────────────────────────────────────────────────
STATE_FEATURES = [
    "stock_ratio",            # stock_on_hand / reorder_point
    "velocity_norm",          # sales_velocity_7d / base_demand
    "days_of_cover_norm",     # days_of_cover / 14 (2-week horizon)
    "stockout_risk",          # probability from ML model [0,1]
    "velocity_trend",         # acceleration/deceleration
    "weather_multiplier",     # external weather impact
    "event_multiplier",       # local event impact
    "promo_multiplier",       # elasticity-driven promo impact
    "day_of_week_sin",        # circular encoding of day
    "day_of_week_cos",
    "is_weekend",
    "days_since_restock_norm", # days_since_restock / lead_time
]
STATE_DIM = len(STATE_FEATURES)


@dataclass
class Experience:
    """A single transition in the replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class InventoryEnvironment:
    """
    Simulated inventory environment for RL training.
    Models the dynamics of a single Sainsbury's product over time.
    """

    def __init__(self, product_data: dict, daily_sales: pd.DataFrame,
                 holding_cost_rate: float = 0.005,
                 stockout_penalty_mult: float = 2.0):
        self.product = product_data
        self.daily_sales = daily_sales.sort_values("date").reset_index(drop=True)
        self.holding_cost_rate = holding_cost_rate
        self.stockout_penalty_mult = stockout_penalty_mult

        self.reorder_point = product_data.get("reorder_point", 20)
        self.unit_price = product_data.get("unit_price", 1.0)
        self.lead_time = product_data.get("lead_time_days", 3)
        self.base_demand = product_data.get("base_demand", 30)

        self.current_step = 0
        self.stock = 0.0
        self.total_reward = 0.0
        self.pending_orders = []  # (delivery_step, quantity)

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode."""
        self.current_step = 0
        self.stock = self.reorder_point * 3  # Start with healthy stock
        self.total_reward = 0.0
        self.pending_orders = []
        return self._get_state()

    def step(self, action_idx: int) -> tuple:
        """
        Take an action (order quantity) and advance one day.

        Returns: (next_state, reward, done, info)
        """
        order_mult = ORDER_ACTIONS[action_idx]
        order_qty = int(order_mult * self.reorder_point)

        # Schedule delivery after lead time
        if order_qty > 0:
            delivery_step = self.current_step + self.lead_time
            self.pending_orders.append((delivery_step, order_qty))

        # Receive any pending deliveries
        received = 0
        remaining = []
        for (d_step, qty) in self.pending_orders:
            if d_step <= self.current_step:
                received += qty
            else:
                remaining.append((d_step, qty))
        self.pending_orders = remaining
        self.stock += received

        # Simulate demand
        if self.current_step < len(self.daily_sales):
            demand = self.daily_sales.iloc[self.current_step].get("units_sold", self.base_demand)
        else:
            demand = self.base_demand * np.random.uniform(0.7, 1.3)

        # Calculate sales and stockout
        actual_sales = min(self.stock, demand)
        stockout_units = max(0, demand - self.stock)
        self.stock = max(0, self.stock - demand)

        # ── Reward calculation ───────────────────────────────────────────────
        # Revenue from sales
        revenue = actual_sales * self.unit_price

        # Holding cost for remaining stock
        holding_cost = self.stock * self.unit_price * self.holding_cost_rate

        # Stockout penalty (lost revenue + customer churn)
        stockout_penalty = stockout_units * self.unit_price * self.stockout_penalty_mult

        # Ordering cost
        order_cost = 0.5 if order_qty > 0 else 0  # Simplified fixed cost

        reward = revenue - holding_cost - stockout_penalty - order_cost

        self.total_reward += reward
        self.current_step += 1

        done = self.current_step >= len(self.daily_sales)

        info = {
            "sales": actual_sales,
            "stockout_units": stockout_units,
            "stock_after": self.stock,
            "order_placed": order_qty,
            "received": received,
            "revenue": revenue,
            "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Extract the state vector from current environment state."""
        step = min(self.current_step, len(self.daily_sales) - 1)
        row = self.daily_sales.iloc[step] if step >= 0 else {}

        velocity = row.get("units_sold", self.base_demand)
        dow = row.get("day_of_week", 0) if isinstance(row, pd.Series) else 0

        state = np.array([
            self.stock / max(self.reorder_point, 1),          # stock_ratio
            velocity / max(self.base_demand, 1),              # velocity_norm
            (self.stock / max(velocity, 0.01)) / 14.0,       # days_of_cover_norm
            row.get("stockout_probability", 0.5) if isinstance(row, pd.Series) else 0.5,
            row.get("velocity_trend", 0.0) if isinstance(row, pd.Series) else 0.0,
            row.get("weather_multiplier", 1.0) if isinstance(row, pd.Series) else 1.0,
            row.get("event_multiplier", 1.0) if isinstance(row, pd.Series) else 1.0,
            row.get("promo_demand_multiplier", 1.0) if isinstance(row, pd.Series) else 1.0,
            np.sin(2 * np.pi * dow / 7),                     # day_of_week_sin
            np.cos(2 * np.pi * dow / 7),                     # day_of_week_cos
            1.0 if dow >= 5 else 0.0,                        # is_weekend
            row.get("days_since_restock", 0) / max(self.lead_time, 1) if isinstance(row, pd.Series) else 0.5,
        ], dtype=np.float32)

        return state


class InventoryRLAgent:
    """
    Q-learning agent with experience replay for inventory replenishment.

    Uses a simple neural-network-free approach (tabular Q with tile coding)
    that's production-ready without PyTorch/TensorFlow dependencies.
    Despite its simplicity, tile-coding Q-learning is surprisingly effective
    for low-dimensional continuous state spaces.
    """

    def __init__(self, n_tilings: int = 8, tiles_per_dim: int = 10,
                 learning_rate: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995):
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.lr = learning_rate / n_tilings  # Divide by tilings for correct averaging
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Tile coding Q-table: maps (tiling_index, tile_indices) → Q-values per action
        self.q_table = {}

        # State space bounds for normalization
        self.state_low = np.array([0, 0, 0, 0, -2, 0.5, 0.5, 0.5, -1, -1, 0, 0])
        self.state_high = np.array([10, 3, 3, 1, 2, 2.0, 2.0, 2.0, 1, 1, 1, 5])

        # Training stats
        self.episode_rewards = []
        self.training_steps = 0

    def _tile_code(self, state: np.ndarray) -> list:
        """
        Convert continuous state to tile-coded features.
        Returns a list of (tiling_idx, tile_hash) tuples.
        """
        # Normalize state to [0, 1]
        norm = np.clip(
            (state - self.state_low) / (self.state_high - self.state_low + 1e-8),
            0, 1
        )

        tiles = []
        for tiling in range(self.n_tilings):
            # Offset each tiling differently
            offset = tiling / self.n_tilings
            indices = tuple(
                int(min((norm[d] + offset) * self.tiles_per_dim, self.tiles_per_dim - 1))
                for d in range(len(state))
            )
            tile_key = (tiling, indices)
            tiles.append(tile_key)

        return tiles

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        tiles = self._tile_code(state)
        q_values = np.zeros(N_ACTIONS)
        for tile in tiles:
            if tile in self.q_table:
                q_values += self.q_table[tile]
        return q_values

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """Update Q-values using TD(0) learning."""
        tiles = self._tile_code(state)

        # Target
        if done:
            target = reward
        else:
            next_q = self._get_q_values(next_state)
            target = reward + self.gamma * np.max(next_q)

        # Current Q-value for this action
        current_q = sum(
            self.q_table.get(tile, np.zeros(N_ACTIONS))[action]
            for tile in tiles
        )

        # TD error
        td_error = target - current_q

        # Update each tiling's contribution
        for tile in tiles:
            if tile not in self.q_table:
                self.q_table[tile] = np.zeros(N_ACTIONS)
            self.q_table[tile][action] += self.lr * td_error

        self.training_steps += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    def train(self, env: InventoryEnvironment, n_episodes: int = 500,
              verbose: bool = True) -> list:
        """
        Train the agent on a simulated environment.

        Returns list of episode rewards.
        """
        logger.info(f"Training RL agent for {n_episodes} episodes...")
        rewards_history = []

        for ep in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            step = 0

            while True:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                self.update(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state
                step += 1

                if done:
                    break

            self.decay_epsilon()
            rewards_history.append(episode_reward)
            self.episode_rewards.append(episode_reward)

            if verbose and (ep + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                logger.info(
                    f"  Episode {ep+1}/{n_episodes} | "
                    f"Avg Reward (50ep): {avg_reward:.1f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        logger.info(
            f"Training complete. Final avg reward: {np.mean(rewards_history[-50:]):.1f} | "
            f"Q-table size: {len(self.q_table)} tiles"
        )
        return rewards_history

    def recommend_action(self, product_state: dict) -> dict:
        """
        Given current product state, recommend an order action.

        Args:
            product_state: dict with keys matching STATE_FEATURES

        Returns:
            dict with recommended_action, order_multiplier, confidence, q_values
        """
        state = self._dict_to_state(product_state)
        q_values = self._get_q_values(state)
        best_action = int(np.argmax(q_values))
        order_mult = ORDER_ACTIONS[best_action]

        # Confidence based on Q-value spread
        q_range = q_values.max() - q_values.min()
        confidence = min(q_range / (abs(q_values.max()) + 1e-6), 1.0)

        reorder_point = product_state.get("reorder_point", 20)
        order_qty = int(order_mult * reorder_point)

        return {
            "action_index": best_action,
            "order_multiplier": order_mult,
            "order_qty": order_qty,
            "q_values": {f"action_{ORDER_ACTIONS[i]}x": round(q_values[i], 2)
                         for i in range(N_ACTIONS)},
            "confidence": round(confidence, 3),
            "agent_recommendation": _action_label(best_action),
        }

    def _dict_to_state(self, d: dict) -> np.ndarray:
        """Convert a product state dict to a numpy state vector."""
        reorder = d.get("reorder_point", 20)
        base_demand = d.get("base_demand", d.get("sales_velocity_7d", 10) * 1.2)
        velocity = d.get("sales_velocity_7d", 10)
        stock = d.get("stock_on_hand", 0)
        dow = d.get("day_of_week", 0)

        return np.array([
            stock / max(reorder, 1),
            velocity / max(base_demand, 1),
            (stock / max(velocity, 0.01)) / 14.0,
            d.get("stockout_probability", 0.5),
            d.get("velocity_trend", 0.0),
            d.get("weather_multiplier", 1.0),
            d.get("event_multiplier", 1.0),
            d.get("promo_demand_multiplier", 1.0),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            1.0 if dow >= 5 else 0.0,
            d.get("days_since_restock", 0) / max(d.get("lead_time_days", 3), 1),
        ], dtype=np.float32)

    def save(self, path: str = RL_MODEL_PATH):
        """Save the trained agent."""
        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episode_rewards": self.episode_rewards[-100:],  # Keep last 100
            "n_tilings": self.n_tilings,
            "tiles_per_dim": self.tiles_per_dim,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"RL agent saved to {path}")

    @classmethod
    def load(cls, path: str = RL_MODEL_PATH) -> "InventoryRLAgent":
        """Load a trained agent."""
        agent = cls()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Reconstruct q_table with proper tuple keys
            agent.q_table = {}
            for k_str, v in data["q_table"].items():
                # Parse string key back to tuple
                key = eval(k_str)  # Safe here as we control the format
                agent.q_table[key] = np.array(v)
            agent.epsilon = data.get("epsilon", 0.05)
            agent.training_steps = data.get("training_steps", 0)
            agent.episode_rewards = data.get("episode_rewards", [])
            logger.info(f"RL agent loaded: {len(agent.q_table)} tiles, "
                        f"{agent.training_steps} training steps")
        else:
            logger.warning(f"No RL agent found at {path}")
        return agent

    def get_training_summary(self) -> dict:
        """Return training statistics."""
        rewards = self.episode_rewards
        if not rewards:
            return {"status": "untrained", "episodes": 0}
        return {
            "status": "trained",
            "episodes": len(rewards),
            "training_steps": self.training_steps,
            "final_avg_reward": round(np.mean(rewards[-50:]), 1),
            "best_episode_reward": round(max(rewards), 1),
            "q_table_size": len(self.q_table),
            "epsilon": round(self.epsilon, 4),
        }


def _action_label(action_idx: int) -> str:
    """Human-readable label for an action."""
    mult = ORDER_ACTIONS[action_idx]
    if mult == 0:
        return "Hold — no order needed"
    elif mult <= 0.5:
        return "Light restock"
    elif mult <= 1.0:
        return "Standard reorder"
    elif mult <= 2.0:
        return "Aggressive restock"
    else:
        return "Emergency bulk order"


def train_rl_agents(sales_df: pd.DataFrame, products_df: pd.DataFrame,
                     n_episodes: int = 300, sample_products: int = 50) -> InventoryRLAgent:
    """
    Train a single RL agent across multiple product environments.
    The agent learns a generalised policy across different product profiles.

    Args:
        sales_df: Historical daily sales data
        products_df: Product metadata
        n_episodes: Training episodes per product
        sample_products: Number of products to train on (for efficiency)

    Returns:
        Trained InventoryRLAgent
    """
    agent = InventoryRLAgent()

    # Sample diverse products for training
    if len(products_df) > sample_products:
        # Stratified sample across categories
        sampled = products_df.groupby("category").apply(
            lambda x: x.sample(min(len(x), max(1, sample_products // x.name.__class__.__hash__(x.name) % 10 + 3)),
                                random_state=42) if len(x) > 0 else x
        ).reset_index(drop=True).head(sample_products)
    else:
        sampled = products_df

    logger.info(f"Training RL agent on {len(sampled)} products × {n_episodes} episodes")

    for idx, (_, product) in enumerate(sampled.iterrows()):
        pid = product["product_id"]
        product_sales = sales_df[sales_df["product_id"] == pid].copy()

        if len(product_sales) < 14:
            continue

        env = InventoryEnvironment(
            product_data=product.to_dict(),
            daily_sales=product_sales,
        )

        agent.train(env, n_episodes=n_episodes, verbose=False)

        if (idx + 1) % 10 == 0:
            avg = np.mean(agent.episode_rewards[-50:]) if agent.episode_rewards else 0
            logger.info(f"  Products trained: {idx+1}/{len(sampled)} | "
                        f"Avg reward: {avg:.1f}")

    agent.save()
    logger.info("RL agent training complete and saved.")
    return agent


if __name__ == "__main__":
    from data_ingestion import load_sales, load_products

    print("=" * 60)
    print("  Phase 4: Reinforcement Learning Agent")
    print("=" * 60)

    sales = load_sales()
    products = load_products()

    print("\n🤖 Training RL Decision Agent...")
    agent = train_rl_agents(sales, products, n_episodes=200, sample_products=20)

    summary = agent.get_training_summary()
    print(f"\n📊 Training Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")

    print("\n🔍 Sample Recommendations:")
    sample_states = [
        {"stock_on_hand": 5, "reorder_point": 30, "sales_velocity_7d": 8,
         "stockout_probability": 0.85, "velocity_trend": 0.2,
         "weather_multiplier": 1.3, "event_multiplier": 1.0,
         "promo_demand_multiplier": 1.0, "day_of_week": 4},
        {"stock_on_hand": 50, "reorder_point": 30, "sales_velocity_7d": 5,
         "stockout_probability": 0.15, "velocity_trend": -0.1,
         "weather_multiplier": 1.0, "event_multiplier": 1.0,
         "promo_demand_multiplier": 1.0, "day_of_week": 1},
    ]
    for state in sample_states:
        rec = agent.recommend_action(state)
        print(f"  Stock={state['stock_on_hand']}, Risk={state['stockout_probability']:.0%} → "
              f"{rec['agent_recommendation']} ({rec['order_qty']} units, "
              f"confidence: {rec['confidence']:.2f})")
