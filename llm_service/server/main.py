import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import asyncio
import grpc
from grpc_reflection.v1alpha import reflection

from common.uhlm import uhlm_pb2, uhlm_pb2_grpc
from .handler import UHLMService

async def serve():
    server = grpc.aio.server()
    uhlm_pb2_grpc.add_UHLMServicer_to_server(UHLMService(model_id="meta-llama/Llama3.1-8B-Instruct"), server)
    
    SERVICE_NAMES = (
        uhlm_pb2.DESCRIPTOR.services_by_name["UHLM"].full_name,
        reflection.SERVICE_NAME
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port("[::]:8081")
    await server.start()
    print("gRPC server started on [::]:8081")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())