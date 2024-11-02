import logging
import os
import sys
from datetime import datetime

class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        
    def write(self, buf):
        # Handle both string and bytes
        if isinstance(buf, bytes):
            buf = buf.decode('utf-8')
            
        # Write to terminal
        self.terminal.write(buf)
        
        # Write to file if content is not empty
        if buf.strip():
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(buf)
    
    def flush(self):
        self.terminal.flush()

def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/session_{timestamp}.log'
    
    # Configure logging with just file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    
    # Replace sys.stdout with our custom logger
    sys.stdout = TeeLogger(log_file)
    
    return log_file, logging.getLogger()