from server import uhlm_pb2, uhlm_pb2_grpc
import asyncio
import grpc
import numpy as np
from transformers import AutoTokenizer  # Add this import


# test_client.py
async def query(prompt: str, max_tokens: int = 20):
    channel = grpc.aio.insecure_channel("localhost:8081")
    stub = uhlm_pb2_grpc.UHLMStub(channel)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    try:
        session_id = (await stub.BeginSession(uhlm_pb2.BeginReq(prompt=prompt))).session_id
        
        # Get prompt token IDs
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        all_tokens = prompt_token_ids.copy()  # Start with prompt tokens
        
        # Generate
        for _ in range(max_tokens):
            resp = await stub.VerifyToken(uhlm_pb2.VerifyReq(
                session_id=session_id,
                draft_id=np.random.randint(0, 32000),
                sparse=uhlm_pb2.SparseTopK(
                    indices=list(range(100, 110)),
                    probs=[0.1] * 10
                )
            ))
            all_tokens.append(resp.token_id)  # Append to full sequence
            if resp.token_id == tokenizer.eos_token_id:  # Use proper EOS token
                break
        
        await stub.EndSession(uhlm_pb2.EndReq(session_id=session_id))
        
        # Decode full sequence (prompt + generated)
        decoded_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
        # Or decode just the generated part
        generated_tokens = all_tokens[len(prompt_token_ids):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return all_tokens, decoded_text, generated_text
        
    finally:
        await channel.close()


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    tokens, decoded, generated = asyncio.run(query(prompt))
    print(f"\nPrompt: {prompt}")
    print(f"Generated tokens: {tokens}")
    print(f"Decoded output: {decoded}")
    print(f"Generated output: {generated}")
