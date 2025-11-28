import numpy as np
import torch
import asyncio
from common.uncertainty import measure_uncertainty_from_logits
from common.rpc_client import UHLMRPCClient
from common.uhlm import uhlm_pb2, uhlm_pb2_grpc
from llm_service.server import verifier # Reuse existing verifier logic
from common.session_manager import SessionManager

class ILMHandler(uhlm_pb2_grpc.UHLMServicer):
    def __init__(self, model_client, llm_host, llm_port, threshold):
        self.model = model_client
        self.tokenizer = model_client.tokenizer
        self.threshold = threshold
        self.sessions = SessionManager(tokenizer=self.tokenizer)
        # Connection to Tier 3 (LLM)
        self.llm_client = UHLMRPCClient(llm_host, llm_port)
        # Map local session IDs to upstream LLM session IDs
        self.session_map = {} 

    async def BeginSession(self, request, context):
        # 1. Start session with upstream LLM
        upstream_sid, eos_id = await self.llm_client.begin_session(request.prompt)
        
        # 2. Initialize local state (SessionManager logic)
        local_sid, _ = self.sessions.begin(request.prompt)
        self.session_map[local_sid] = upstream_sid
        
        return uhlm_pb2.BeginResp(session_id=local_sid, eos_token_id=eos_id)

    async def VerifyToken(self, request, context):
        # 1. Get context and run ILM Inference
        text = self.sessions.get_text(request.session_id)
        
        # Get probabilities from VLLMClient (returns numpy array of shape [vocab_size])
        ilm_probs_np = await self.model.logits(text)
        
        # Convert to torch tensor for uncertainty measurement
        # VLLMClient returns softmaxed probs. We approximate logits by taking log.
        # Add epsilon to avoid log(0).
        ilm_logits = torch.tensor(np.log(ilm_probs_np + 1e-10))
        
        # 2. Calculate ILM Uncertainty
        u_ilm, _ = measure_uncertainty_from_logits(ilm_logits)
        
        # 3. Decision Logic
        if u_ilm > self.threshold:
            # --- CASE A: ILM is Uncertain -> Offload to LLM ---
            upstream_sid = self.session_map[request.session_id]
            
            # Forward the verification request to the LLM
            # We pass the SLM's original draft and probs directly (forwarding the proto message)
            if self.llm_client.simulate_latency:
                await asyncio.sleep(self.llm_client.latency_seconds)


            which = request.WhichOneof("slm_probs")
            if not which:
                raise ValueError("VerifyToken missing slm_probs")
            slm_payload = getattr(request, which)

            resp = await self.llm_client.stub.VerifyToken(
                uhlm_pb2.VerifyReq(
                    session_id=upstream_sid,
                    draft_id=request.draft_id,
                    # sparse=request.slm_probs
                    **{which: slm_payload}
                )
            )
            accepted = resp.accepted
            token_id = resp.token_id
            
            # Update local state with whatever the LLM decided
            self.sessions.append(request.session_id, token_id)
            return uhlm_pb2.VerifyResp(accepted=accepted, token_id=token_id)
            
        else:
            # --- CASE B: ILM is Confident -> Verify Locally ---
            # Use ILM's probs (ilm_probs_np) as ground truth (y)
            # Use SLM's probs (request.slm_probs) as draft (x)
            
            which = request.WhichOneof("slm_probs")
            if not which:
                raise ValueError("VerifyToken missing slm_probs")
            slm_payload = getattr(request, which)

            if which == "sparse":
                slm_probs = self._unpack_sparse(slm_payload)
            else:
                slm_probs = np.array(slm_payload.probs, dtype=np.float32)
                total = slm_probs.sum()
                if total: 
                    slm_probs /= total

            accepted, token_id = verifier.accept_or_resample(
                request.draft_id, slm_probs, ilm_probs_np
            )
            
            # If we verified locally, we must asynchronously sync the LLM 
            # so it doesn't fall behind (optional but recommended for consistency)
            upstream_sid = self.session_map[request.session_id]
            # asyncio.create_task(self.llm_client.sync(upstream_sid, [token_id]))

            loop = asyncio.get_event_loop()
            loop.create_task(self.llm_client.sync(upstream_sid, [token_id]))
            
            self.sessions.append(request.session_id, token_id)
            return uhlm_pb2.VerifyResp(accepted=accepted, token_id=token_id)

    async def Sync(self, request, context):
        # Update local state
        self.sessions.sync_tail(request.session_id, request.tail_ids)
        
        # Forward sync to LLM
        upstream_sid = self.session_map[request.session_id]
        await self.llm_client.sync(upstream_sid, request.tail_ids)
        
        return uhlm_pb2.SyncResp()

    def _unpack_sparse(self, sparse_proto):
        """Reconstruct dense probability array from sparse proto."""
        vocab_size = self.model.vocab_size
        probs = np.zeros(vocab_size, dtype=np.float32)
        
        indices = sparse_proto.indices
        values = sparse_proto.probs
        
        # Scatter values
        probs[indices] = values
        
        # Re-normalize in case of truncation or precision issues
        # (Though strictly speaking, we treat unlisted as 0)
        s = np.sum(probs)
        if s > 0:
            probs /= s
            
        return probs


    async def EndSession(self, request, context):
        ok = self.sessions.end(request.session_id)
        return uhlm_pb2.EndResp(success=ok)