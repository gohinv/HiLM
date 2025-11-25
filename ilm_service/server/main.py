import sys
import asyncio
import argparse
from pathlib import Path
from concurrent import futures
import grpc

# Add repository root to Python path so we can import common/
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.uhlm import uhlm_pb2_grpc
from handler import ILMHandler  # You would create this handler
from llm_service.server.vllm_client import VLLMClient # Reuse VLLM client if possible or duplicate

MODEL_ID = "meta-llama/Llama3.2-3B-Instruct"
LLM_HOST = "127.0.0.1"
LLM_PORT = 8081
THRESHOLD = 0.4
TENSOR_PARALLEL_SIZE = 1
PORT = 8082

def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Initialize the local model (The "Intermediate" model)
    print(f"Loading ILM Model: {MODEL_ID}")
    # distinct tensor_parallel_size for ILM vs LLM if on same machine
    local_model_client = VLLMClient(model_id=MODEL_ID, tensor_parallel_size=TENSOR_PARALLEL_SIZE)
    
    # Initialize the Handler
    # This handler needs to know where the upstream LLM is
    handler = ILMHandler(
        model_client=local_model_client,
        llm_host=LLM_HOST,
        llm_port=LLM_PORT,
        threshold=THRESHOLD
    )
    
    uhlm_pb2_grpc.add_UHLMServicer_to_server(handler, server)
    
    listen_addr = f'[::]:{PORT}'
    server.add_insecure_port(listen_addr)
    print(f"✅ ILM Service started on {listen_addr}")
    print(f"   └── Upstream LLM: {LLM_HOST}:{LLM_PORT}")
    print(f"   └── Uncertainty Threshold: {THRESHOLD}")
    
    async def server_loop():
        await server.start()
        await server.wait_for_termination()
        
    asyncio.run(server_loop())
 
if __name__ == '__main__':
    serve()