# common/llm_rpc_client.py
import sys
from pathlib import Path
import asyncio

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]  # common -> U-HLM
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import grpc
from common.uhlm import uhlm_pb2, uhlm_pb2_grpc # adjust import path
import numpy as np

class UHLMRPCClient:
    def __init__(self, host="127.0.0.1", port=8081, simulate_latency=False, latency_ms=50):
        self.channel = grpc.aio.insecure_channel(f"{host}:{port}")
        self.stub = uhlm_pb2_grpc.UHLMStub(self.channel)
        self.simulate_latency = simulate_latency
        self.latency_seconds = latency_ms / 1000.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.channel.close()

    async def begin_session(self, prompt):
        if self.simulate_latency:
            await asyncio.sleep(self.latency_seconds)
        resp = await self.stub.BeginSession(uhlm_pb2.BeginReq(prompt=prompt))
        return resp.session_id, resp.eos_token_id


    async def verify(self, session_id, draft_id, probs):
        if self.simulate_latency:
            await asyncio.sleep(self.latency_seconds)
        vocab_size = len(probs) # probs is [vocab_size] array, get top-K actual token IDs

        # Get top-K token IDs (indices where probs are highest)
        top_k = min(100, vocab_size)  # Limit to top 100
        top_indices = np.argsort(probs)[-top_k:][::-1]  # Descending order
        top_probs = probs[top_indices]
        
        # Normalize probabilities
        top_probs = top_probs / np.sum(top_probs)
        
        sparse = uhlm_pb2.SparseTopK(
            indices=top_indices.astype(np.uint32).tolist(),  # Actual token IDs
            probs=top_probs.astype(np.float32).tolist(),
        )
        resp = await self.stub.VerifyToken(
            uhlm_pb2.VerifyReq(session_id=session_id,
                            draft_id=draft_id,
                            sparse=sparse)
        )
        return resp.accepted, resp.token_id, resp.new_length

    async def sync(self, session_id, tail_ids):
        if self.simulate_latency:
            await asyncio.sleep(self.latency_seconds)
        resp = await self.stub.Sync(
            uhlm_pb2.SyncReq(session_id=session_id, tail_ids=[int(t) for t in tail_ids])
        )
        return resp.new_length

    async def end_session(self, session_id):
        if self.simulate_latency:
            await asyncio.sleep(self.latency_seconds)
        await self.stub.EndSession(uhlm_pb2.EndReq(session_id=session_id))