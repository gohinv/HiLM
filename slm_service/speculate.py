import requests
import json
import torch
import torch.nn.functional as F
import asyncio
from common.rpc_client import UHLMRPCClient

def sample_draft_tokens(model, context, K=20, theta_max=2.0, device="cuda"):
    """
    Run the SLM (e.g. Llama3.2 1B) once to get logits, then perform K temperature perturbations.
    Returns:
        - base_draft_id: the draft token from the original distribution (θ=1)
        - base_probs: the base softmax distribution (x(t))
        - sampled_ids: list of K sampled tokens from perturbed temperatures
        - thetas: list of the θ values used
    """
    model.eval()
    context = context.to(device)
    with torch.no_grad():
        logits = model(context).logits
        next_logits = logits[:, -1, :]  # [1, vocab_size]
        base_probs = F.softmax(next_logits, dim=-1).squeeze(0)

    # Base draft (no perturbation)
    base_draft_id = torch.multinomial(base_probs, 1).item()

    # Generate K temperature samples
    thetas = torch.linspace(0.0, theta_max, K).tolist()
    sampled_ids = []
    for theta in thetas:
        if theta == 0:
            # handle division by 0 case (θ=0 means deterministic argmax)
            perturbed_probs = torch.zeros_like(base_probs)
            perturbed_probs[torch.argmax(base_probs)] = 1.0
        else:
            perturbed_probs = F.softmax(next_logits / theta, dim=-1).squeeze(0)
        sampled_id = torch.multinomial(perturbed_probs, 1).item()
        sampled_ids.append(sampled_id)

    return {
        "base_draft_id": base_draft_id,
        "base_probs": base_probs,
        "sampled_ids": sampled_ids,
        "thetas": thetas
    }


async def adaptive_offload(prompt, session_id, draft_id, probs):
    async with UHLMRPCClient() as llm:
        accepted, token_id, new_length = await llm.verify(
            session_id=session_id,
            draft_id=draft_id,
            probs=probs
        )
        return token_id, accepted
