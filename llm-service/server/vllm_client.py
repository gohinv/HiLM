import numpy as np
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs

class VLLMClient:
    def __init__(self, model_id="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1):
        # Initialize vLLM engine
        engine_args = EngineArgs(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.tokenizer = self.engine.tokenizer
        # Get vocab size
        try:
            self.vocab_size = len(self.tokenizer.get_vocab())
        except:
            try:
                self.vocab_size = self.engine.llm_engine.model_config.get_vocab_size()
            except:
                self.vocab_size = 32000  # Default for Llama
        
    async def logits(self, session_text):
        """
        Get logits for next token given session text.
        Returns softmax-normalized probs as numpy array.
        """
        # Create request
        request_id = f"req_{id(session_text)}"
        
        try:
            self.engine.add_request(
                request_id=request_id,
                prompt=session_text,
                params=SamplingParams(temperature=1.0, logprobs=1)
            )
        except ValueError as e:
            print(f"Error adding request {request_id}: {e}")
            return None

        final_output = None

        while True:
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                if request_output.request_id == request_id:
                    final_output = request_output
                    if final_output.finished:
                        break

            if not self.engine.has_unfinished_requests():
                break

        # Extract logits and convert to probabilities
        if final_output and final_output.outputs and final_output.outputs[0].logprobs:
            # logprobs is a list of dicts, where each dict maps token_id -> Logprob object
            logprobs_list = final_output.outputs[0].logprobs
            
            if not logprobs_list:
                # Fallback to uniform
                probs = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size
                return probs
            
            # Get the first token's logprobs (dict of token_id -> Logprob)
            logprobs_dict = logprobs_list[0]
            
            # Create array of logits (initialize with -inf)
            logits = np.full(self.vocab_size, -np.inf, dtype=np.float32)
            
            # Extract float values from Logprob objects
            for token_id, logprob_obj in logprobs_dict.items():
                if 0 <= token_id < self.vocab_size:
                    # Extract the actual logprob value from the Logprob object
                    if hasattr(logprob_obj, 'logprob'):
                        logits[token_id] = float(logprob_obj.logprob)
                    elif hasattr(logprob_obj, '__float__'):
                        logits[token_id] = float(logprob_obj)
                    elif isinstance(logprob_obj, (int, float)):
                        logits[token_id] = float(logprob_obj)
                    else:
                        # Try to access as attribute or dict
                        try:
                            logits[token_id] = float(logprob_obj)
                        except:
                            print(f"Warning: Could not extract logprob for token {token_id}: {type(logprob_obj)}")
                            continue
            
            # Convert logits to probabilities (softmax)
            # Replace -inf with very small number for numerical stability
            logits = np.where(logits == -np.inf, -1e10, logits)
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            
            return probs.astype(np.float32)
        else:
            # Fallback: return uniform distribution
            print("Warning: No logprobs returned, using uniform distribution")
            probs = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size
            return probs