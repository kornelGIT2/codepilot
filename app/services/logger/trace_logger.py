import json 
import os 
from datetime import datetime

class TraceLogger:
    def __init__(self, log_dir="logs/traces"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def log(self, data: dict):
        filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        file_path = os.path.join(self.log_dir, filename)
        self._save_logs(file_path, data)

    def _save_logs(self, file_path, data):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)