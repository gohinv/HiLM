Run the SLM-service: 

```
python3 -m venv venv
pip3 -r requirements.txt

# run script (connects to localhost LLM)

# to simulate latency, add the --latency flag
# to enable chat-template, add the --use-chat-template flag
python3 main.py  
```
