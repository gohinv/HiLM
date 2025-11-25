import uuid

class SessionManager:
    def __init__(self, tokenizer=None):
        self.sessions = {}
        self.tokenizer = tokenizer
    
    def begin(self, prompt):
        sid = str(uuid.uuid4())
        # Tokenize the prompt and store token IDs
        if self.tokenizer:
            prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            # Store both original text and token IDs
            self.sessions[sid] = {
                "prompt": prompt,
                "token_ids": prompt_token_ids.copy(),  # Store token IDs
                "text": prompt,  # Keep original for display
                "length": len(prompt_token_ids),
            }
        else:
            self.sessions[sid] = {
                "prompt": prompt,
                "token_ids": [],
                "text": prompt,
                "length": len(prompt.split()),
            }
        return sid, self.sessions[sid]["length"]
    
    def append(self, sid, token_id):
        """Append token ID"""
        self.sessions[sid]["token_ids"].append(token_id)
        self.sessions[sid]["length"] += 1
    
    def get_text(self, sid):
        """Get text for vLLM - reconstruct from token IDs"""
        session = self.sessions[sid]
        if self.tokenizer and session["token_ids"]:
            # Decode token IDs back to text for vLLM
            return self.tokenizer.decode(session["token_ids"], skip_special_tokens=False)
        return session["text"]
    
    def get_token_ids(self, sid):
        return self.sessions[sid]["token_ids"]
    
    def sync_tail(self, sid, tail):
        self.sessions[sid]["token_ids"].extend(tail)
        self.sessions[sid]["length"] += len(tail)
    
    def end(self, sid):
        if sid in self.sessions:
            del self.sessions[sid]
            return True
        return False

