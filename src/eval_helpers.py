import torch
import numpy as np
from .memory import Episode
from tqdm import tqdm


def baseline_prediction(
    transition_model: torch.nn.Module,
    episodes: list[Episode],
    warmup: int = 0,
    unroll: int = 1,
    max_steps: int = 10,
) -> dict:
    device = transition_model.device
    transition_model.eval()

    # get the states, actions, and next states
    states = torch.stack(
        [torch.stack([step.state for step in episode]) for episode in episodes], dim=1
    ).to(device)
    actions = torch.stack(
        [torch.stack([step.action for step in episode]) for episode in episodes], dim=1
    ).to(device)
    next_states = torch.stack(
        [torch.stack([step.next_state for step in episode]) for episode in episodes],
        dim=1,
    ).to(device)

    # make 1 step predictions
    transition_model.reset_state()
    delta_states = transition_model(states, actions, deterministic=True)
    predicted_next_states = states + delta_states
    predicted_state_mse = torch.nn.functional.mse_loss(
        predicted_next_states[warmup:], next_states[warmup:]
    )

    # make unrolled predictions
    steps = states.shape[0] - warmup - unroll
    steps = min(steps, max_steps)
    unrolled_state_mse = torch.zeros(1, device=device)
    if steps > 0:
        for step in range(steps):
            transition_model.reset_state()
            _ = transition_model(
                states[step : warmup + step],
                actions[step : warmup + step],
                deterministic=True,
            )
            pred_states = torch.empty((unroll, *states.shape[1:]), device=device)
            state = states[warmup + step : warmup + step + 1]
            for i in range(unroll):
                action = actions[warmup + step + i : warmup + step + i + 1]
                delta_state = transition_model(state, action, deterministic=True)
                pred_states[i] = state + delta_state.detach()
                state = pred_states[i]
            unrolled_state_mse += torch.nn.functional.mse_loss(
                pred_states, next_states[warmup + step : warmup + step + unroll]
            )
        unrolled_state_mse /= steps

    # make linear extrapolations
    previous_deltas = states[1:] - states[:-1]
    extrapolated_next_states = states[1:] + previous_deltas
    extrapolated_state_mse = torch.nn.functional.mse_loss(
        extrapolated_next_states[warmup:], next_states[1 + warmup :]
    )

    # current state mse
    current_state_mse = torch.nn.functional.mse_loss(
        states[warmup:], next_states[warmup:]
    )

    return {
        "predicted state mse": predicted_state_mse.item(),
        "unrolled state mse": unrolled_state_mse.item(),
        "extrapolated state mse": extrapolated_state_mse.item(),
        "current state mse": current_state_mse.item(),
    }


def make_predictions(
    transition_model: torch.nn.Module,
    episodes: list[Episode],
    warmup: int = 0,
    unroll: int = 1,
    step: int = 10,
    deterministic: bool = True,
) -> dict:
    assert warmup >= 1

    device = transition_model.device
    transition_model.eval()

    # get the states, actions, and next states
    states = torch.stack(
        [torch.stack([s.state for s in episode]) for episode in episodes], dim=1
    ).to(device)
    actions = torch.stack(
        [torch.stack([s.action for s in episode]) for episode in episodes], dim=1
    ).to(device)

    T = states.shape[0]

    predictions = torch.zeros(
        (T // step, warmup + unroll, *states.shape[1:]), device=device
    )

    pbar = tqdm(np.arange(0, T, step), desc=f"{'rolling out state predictions':30}")
    for t in pbar:
        # make unrolled predictions
        transition_model.reset_state()
        for j in range(warmup + unroll):
            if t + j >= T:
                break
            if j < warmup:
                state = states[t + j]
            else:
                state = next_state_hat
            state_delta_hat = transition_model(
                state, actions[t + j], deterministic=deterministic
            )
            next_state_hat = state + state_delta_hat
            predictions[t // step, j] = next_state_hat.detach()

    return predictions
