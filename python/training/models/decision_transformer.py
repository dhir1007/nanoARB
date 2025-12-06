"""Decision Transformer for offline RL market-making."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal self-attention for Decision Transformer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer for market-making.

    Conditions on returns-to-go, states, and actions to predict next action.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_timesteps: int = 1000,
        context_length: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.return_embed = nn.Linear(1, hidden_dim)

        # Timestep embedding
        self.timestep_embed = nn.Embedding(max_timesteps, hidden_dim)

        # Position embedding for sequence
        self.pos_embed = nn.Embedding(3 * context_length, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            states: State observations (batch, context_len, state_dim)
            actions: Actions taken (batch, context_len, action_dim)
            returns_to_go: Future returns (batch, context_len, 1)
            timesteps: Timestep indices (batch, context_len)

        Returns:
            Predicted actions (batch, context_len, action_dim)
        """
        batch, seq_len = states.shape[:2]

        # Get embeddings
        state_emb = self.state_embed(states)
        action_emb = self.action_embed(actions)
        return_emb = self.return_embed(returns_to_go)

        # Add timestep embeddings
        time_emb = self.timestep_embed(timesteps)
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        return_emb = return_emb + time_emb

        # Interleave: [R1, S1, A1, R2, S2, A2, ...]
        # Shape: (batch, 3 * seq_len, hidden_dim)
        stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
        stacked = stacked.view(batch, 3 * seq_len, self.hidden_dim)

        # Add positional embeddings
        positions = torch.arange(3 * seq_len, device=states.device)
        stacked = stacked + self.pos_embed(positions)

        # Transformer blocks
        for block in self.blocks:
            stacked = block(stacked)

        stacked = self.norm(stacked)

        # Extract state positions (indices 1, 4, 7, ... i.e., 3k+1)
        state_outputs = stacked[:, 1::3, :]

        # Predict actions
        action_preds = self.action_head(state_outputs)

        return action_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get single action for inference.

        Takes context history and returns next action.
        """
        # Ensure we don't exceed context length
        states = states[:, -self.context_length:]
        actions = actions[:, -self.context_length:]
        returns_to_go = returns_to_go[:, -self.context_length:]
        timesteps = timesteps[:, -self.context_length:]

        # Pad if necessary
        seq_len = states.shape[1]
        if seq_len < self.context_length:
            pad_len = self.context_length - seq_len
            states = F.pad(states, (0, 0, pad_len, 0))
            actions = F.pad(actions, (0, 0, pad_len, 0))
            returns_to_go = F.pad(returns_to_go, (0, 0, pad_len, 0))
            timesteps = F.pad(timesteps, (pad_len, 0))

        action_preds = self.forward(states, actions, returns_to_go, timesteps)

        # Return last action prediction
        return action_preds[:, -1, :]


class ImplicitQLearning(nn.Module):
    """Implicit Q-Learning (IQL) for offline RL market-making."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        expectile: float = 0.7,
        temperature: float = 3.0,
    ):
        super().__init__()
        self.expectile = expectile
        self.temperature = temperature

        # Value network V(s)
        self.value_net = self._build_mlp(state_dim, hidden_dim, 1, num_layers)

        # Q networks Q(s, a) - using twin Q for stability
        self.q1_net = self._build_mlp(state_dim + action_dim, hidden_dim, 1, num_layers)
        self.q2_net = self._build_mlp(state_dim + action_dim, hidden_dim, 1, num_layers)

        # Policy network
        self.policy_net = self._build_mlp(state_dim, hidden_dim, action_dim, num_layers)

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> nn.Sequential:
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        """Get state values V(s)."""
        return self.value_net(states)

    def get_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get Q values Q(s, a) from both networks."""
        sa = torch.cat([states, actions], dim=-1)
        return self.q1_net(sa), self.q2_net(sa)

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        """Get action from policy."""
        return torch.tanh(self.policy_net(states))

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
    ) -> dict[str, torch.Tensor]:
        """Compute IQL losses.

        Returns dict with value_loss, q_loss, and policy_loss.
        """
        with torch.no_grad():
            # Target value for Q update
            next_v = self.get_value(next_states)
            q_target = rewards + gamma * (1 - dones) * next_v

        # Q losses
        q1, q2 = self.get_q_values(states, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Value loss (expectile regression)
        with torch.no_grad():
            q = torch.min(q1, q2)
        v = self.get_value(states)
        diff = q - v
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        value_loss = (weight * diff.pow(2)).mean()

        # Policy loss (advantage weighted regression)
        with torch.no_grad():
            adv = q - v
            weights = torch.exp(adv * self.temperature)
            weights = torch.clamp(weights, max=100.0)

        pred_actions = self.get_action(states)
        policy_loss = (weights * (pred_actions - actions).pow(2).sum(dim=-1)).mean()

        return {
            "value_loss": value_loss,
            "q_loss": q_loss,
            "policy_loss": policy_loss,
            "total_loss": value_loss + q_loss + policy_loss,
        }


if __name__ == "__main__":
    # Test Decision Transformer
    dt = DecisionTransformer(
        state_dim=50,
        action_dim=5,
        hidden_dim=128,
        num_layers=4,
        context_length=20,
    )

    batch = 2
    ctx_len = 20

    states = torch.randn(batch, ctx_len, 50)
    actions = torch.randn(batch, ctx_len, 5)
    returns = torch.randn(batch, ctx_len, 1)
    timesteps = torch.arange(ctx_len).unsqueeze(0).expand(batch, -1)

    action_preds = dt(states, actions, returns, timesteps)
    print(f"DT output shape: {action_preds.shape}")

    # Test IQL
    iql = ImplicitQLearning(state_dim=50, action_dim=5)

    states = torch.randn(32, 50)
    actions = torch.randn(32, 5)
    rewards = torch.randn(32, 1)
    next_states = torch.randn(32, 50)
    dones = torch.zeros(32, 1)

    losses = iql.compute_loss(states, actions, rewards, next_states, dones)
    print(f"IQL losses: {losses}")

