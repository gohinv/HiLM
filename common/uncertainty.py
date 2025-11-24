import torch
import torch.nn.functional as F

def measure_uncertainty_from_logits(logits, K=20, theta_max=2.0):
    """
    Generic function to calculate uncertainty from logits using temp perturbation.
    Used by both SLM (generation) and ILM (verification).
    """
    base_probs = F.softmax(logits, dim=-1)
    base_draft_id = torch.argmax(base_probs).item() # Or sample

    thetas = torch.linspace(0.0, theta_max, K).tolist()
    sampled_ids = []
    
    for theta in thetas:
        if theta == 0:
            p = torch.zeros_like(base_probs)
            p[torch.argmax(base_probs)] = 1.0
        else:
            p = F.softmax(logits / theta, dim=-1)
        
        sampled_ids.append(torch.multinomial(p, 1).item())

    disagreements = sum(1 for s in sampled_ids if s != base_draft_id)
    u_t = disagreements / len(sampled_ids)
    
    return u_t, base_probs