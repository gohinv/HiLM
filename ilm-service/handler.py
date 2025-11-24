import numpy as np
import torch
from common.uncertainty import measure_uncertainty_from_logits
from common.rpc_client import UHLMRPCClient
from common.uhlm import uhlm_pb2, uhlm_pb2_grpc
from llm_service.server import verifier # Reuse existing verifier logic

class ILMHandler(uhlm_pb2_grpc.UHLMServicer):
    def __init__(self, model, tokenizer, llm_host, llm_port, threshold):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        # Connection to Tier 3 (LLM)
        self.llm_client = UHLMRPCClient(llm_host, llm_port)
        # Map local session IDs to upstream LLM session IDs
        self.session_map = {} 

    async def BeginSession(self, request, context):
        # 1. Start session with upstream LLM
        upstream_sid, eos_id = await self.llm_client.begin_session(request.prompt)
        
        # 2. Initialize local state (SessionManager logic)
        local_sid = self.sessions.create(request.prompt)
        self.session_map[local_sid] = upstream_sid
        
        return uhlm_pb2.BeginResp(session_id=local_sid, eos_token_id=eos_id)

    async def VerifyToken(self, request, context):
        # 1. Get context and run ILM Inference
        input_ids = self.sessions.get_tokens(request.session_id)
        logits = self.run_model(input_ids) # Your model inference
        
        # 2. Calculate ILM Uncertainty
        u_ilm, ilm_probs = measure_uncertainty_from_logits(logits)
        
        # 3. Decision Logic
        if u_ilm > self.threshold:
            # --- CASE A: ILM is Uncertain -> Offload to LLM ---
            upstream_sid = self.session_map[request.session_id]
            
            # Forward the verification request to the LLM
            # We pass the SLM's original draft and probs
            accepted, token_id, new_len = await self.llm_client.verify(
                upstream_sid, 
                request.draft_id, 
                request.slm_probs # You'll need to unpack/repack this
            )
            
            # Update local state with whatever the LLM decided
            self.sessions.append(request.session_id, token_id)
            return uhlm_pb2.VerifyResp(accepted=accepted, token_id=token_id)
            
        else:
            # --- CASE B: ILM is Confident -> Verify Locally ---
            # Use ILM's probs (ilm_probs) as ground truth (y)
            # Use SLM's probs (request.slm_probs) as draft (x)
            
            slm_probs = self._unpack_sparse(request.slm_probs)
            accepted, token_id = verifier.accept_or_resample(
                request.draft_id, slm_probs, ilm_probs.numpy()
            )
            
            # If we verified locally, we must asynchronously sync the LLM 
            # so it doesn't fall behind (optional but recommended for consistency)
            upstream_sid = self.session_map[request.session_id]
            asyncio.create_task(self.llm_client.sync(upstream_sid, [token_id]))
            
            self.sessions.append(request.session_id, token_id)
            return uhlm_pb2.VerifyResp(accepted=accepted, token_id=token_id)

    async def Sync(self, request, context):
        # Update local state
        self.sessions.extend(request.session_id, request.tail_ids)
        
        # Forward sync to LLM
        upstream_sid = self.session_map[request.session_id]
        await self.llm_client.sync(upstream_sid, request.tail_ids)
        
        return uhlm_pb2.SyncResp()