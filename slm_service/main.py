import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]  # slm-service -> U-HLM
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np
import threshold_calc
import speculate
from  common.rpc_client import UHLMRPCClient
import asyncio
import utils
import argparse

from transformers import AutoTokenizer


# For Q&A with base models, use Q: A: formatting
model, tokenizer = utils.setup("meta-llama/Llama3.2-1B-Instruct")
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

async def generate_response(prompt, max_tokens=50, K=20, theta_max=2.0, use_chat_template=False, simulate_network=False):
    """Generate a complete response using U-HLM with gRPC LLM verification."""
    print(f"\nGenerating response for: '{prompt}'")
    print("-" * 60)

    if simulate_network:
        print("⚠️  Network latency simulation enabled (50ms per RPC call)")

    # Format prompt for chat model if enabled and tokenizer has chat template
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Using chat template. Formatted prompt: {repr(formatted_prompt)}")
    else:
        # For base models: format questions as Q: A: for better completion
        if prompt.strip().endswith('?'):
            formatted_prompt = f"Q: {prompt}\nA:"
            print(f"Base model Q&A format: {repr(formatted_prompt)}")
        else:
            formatted_prompt = prompt
            print("Using raw prompt")

    # Tokenize prompt and remove EOS tokens
    current_token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    slm_eos_id = tokenizer.eos_token_id
    current_token_ids = [t for t in current_token_ids if t != slm_eos_id]
    
    if not current_token_ids:
        print("Prompt contains only EOS tokens, stopping.")
        return "", {"transmitted": 0, "skipped": 0, "total": 0, "transmission_rate": 0.0}
    
    response_token_ids = []
    transmitted_count = 0
    skipped_count = 0

    async with UHLMRPCClient(host="127.0.0.1", port=8081, simulate_latency=simulate_network) as llm, UHLMRPCClient(host="127.0.0.1", port=8082, simulate_latency=simulate_network) as ilm:
        # Get session ID and LLM's EOS token ID (use formatted prompt for both SLM and LLM)
        session_id, llm_eos_token_id = await llm.begin_session(formatted_prompt)
        ilm_session_id, ilm_eos_token_id = await ilm.begin_session(formatted_prompt)
        print(f"LLM EOS token ID: {llm_eos_token_id}")
        print(f"ILM EOS token ID: {ilm_eos_token_id}")
        print(f"SLM EOS token ID: {slm_eos_id}")

        try:
            for step in range(max_tokens):
                # 2. Run the SLM to propose a draft token + distribution
                context_tensor = torch.tensor([current_token_ids]).to(model.device)
                draft = speculate.sample_draft_tokens(
                    model, context_tensor, K=K, theta_max=theta_max, device=model.device
                )

                base_draft_id = draft["base_draft_id"]
                base_probs = draft["base_probs"].detach().cpu().numpy()

                # 3. Measure uncertainty
                sampled_ids = draft["sampled_ids"]
                u_t = sum(d_k != base_draft_id for d_k in sampled_ids) / len(sampled_ids)
                t1, t2 = threshold_calc.get_threshold()

                final_token_id = None
                decision = ""

                # skip all transmission
                if u_t < t1:
                    decision = "SKIP"
                    final_token_id = base_draft_id
                    
                    # update contexts of both upstream models
                    await asyncio.gather(
                        ilm.sync(ilm_session_id, [final_token_id]),
                        llm.sync(session_id, [final_token_id])
                    )

                # verify with ILM
                elif u_t < t2:
                    decision = "VERIFY_ILM"
                    
                    accepted, final_token_id, _ = await ilm.verify(
                        ilm_session_id, base_draft_id, base_probs
                    )
                    
                    # update context of LLM with the result from ILM
                    await llm.sync(session_id, [final_token_id])

                # verify with LLM
                else:
                    decision = "VERIFY_LLM"
                    
                    accepted, final_token_id, _ = await llm.verify(
                        session_id, base_draft_id, base_probs
                    )
                    
                    # update context of ILM with the result from LLM
                    await ilm.sync(ilm_session_id, [final_token_id])


                # check EOS BEFORE appending
                if decision == "VERIFY_ILM":
                    # Token came from LLM, use LLM's EOS token ID
                    if int(final_token_id) == int(ilm_eos_token_id):
                        print(f"Hit ILM EOS token (id={final_token_id}) after ILM verification; stopping generation.")
                        break
                elif decision == "VERIFY_LLM":
                    # Token came from LLM, use LLM's EOS token ID
                    if int(final_token_id) == int(llm_eos_token_id):
                        print(f"Hit LLM EOS token (id={final_token_id}) after LLM verification; stopping generation.")
                        break
                else:
                    # Token came from SLM, use SLM's EOS token ID
                    if int(final_token_id) == int(slm_eos_id):
                        print(f"Hit SLM EOS token (id={final_token_id}) after skipping; stopping generation.")
                        break

                # 6. Append token (only if not EOS)
                current_token_ids.append(final_token_id)
                response_token_ids.append(final_token_id)

                # 7. Log the choice
                token_text = tokenizer.decode([final_token_id], skip_special_tokens=True).strip()
                if final_token_id == tokenizer.eos_token_id:
                    print(f"[SLM DEBUG] Generated EOS token: final_token_id={final_token_id}, SLM eos_token_id={tokenizer.eos_token_id}")
                display_text = token_text if token_text else f'<EOS or empty, id={final_token_id}>'
                
                # Determine which threshold was relevant
                relevant_thresh = t1 if decision == "SKIP" else t2
                print(
                    f"Token {step+1:>3}: [{decision:<10}] "
                    f"u={u_t:.3f} vs T={relevant_thresh:.3f} "
                    f"{'accepted' if decision != 'SKIP' and accepted else 'skipped ' if decision == 'SKIP' else 'rejected'} -> '{display_text}'"
                )

        finally:
            # 8. Cleanly close the session
            await llm.end_session(session_id)
            await ilm.end_session(session_id)

    # 9. Decode and report statistics
    decoded = tokenizer.decode(response_token_ids, skip_special_tokens=True)
    total = len(response_token_ids) or 1
    print(f"\nComplete Response:")
    print(decoded if decoded.strip() else "<empty>")
    print(
        f"\nStats: transmitted={transmitted_count}, skipped={skipped_count}, "
        f"total={total}, transmission_rate={(transmitted_count/total)*100:.1f}%"
    )

    return decoded, {
        "transmitted": transmitted_count,
        "skipped": skipped_count,
        "total": total,
        "transmission_rate": transmitted_count / total if total else 0.0,
    }


# Parse command-line arguments
parser = argparse.ArgumentParser(description='U-HLM: Uncertainty-Aware Hybrid Language Model Inference')
parser.add_argument('--latency', '--simulate-latency', action='store_true',
                    help='Simulate 50ms network latency for RPC calls (default: False)')
parser.add_argument('--use-chat-template', action='store_true',
                    help='Use chat template formatting for prompts (default: False)')
args = parser.parse_args()

def main():
    """Main inference loop"""
        
    while True:
        prompt = input("\nEnter prompt (or 'q'/'quit' to exit): ").strip()
        
        if prompt.lower() == 'quit' or prompt.lower() == 'q':
            break
        
        if not prompt:
            continue
            
        try:
            asyncio.run(generate_response(prompt, use_chat_template=args.use_chat_template, simulate_network=args.latency))
        except Exception as e:
            print(f"Error generating response: {e}")
            continue

if __name__ == "__main__":
    main()