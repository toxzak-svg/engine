import os
from pathlib import Path
from datetime import datetime

# Define the base path and today's date
base_path = 'c:/dev/engine/artifacts'
today = datetime.now().strftime('%Y-%m-%d')

# Dictionary to track file types
file_types = {}

# Walk through the directory
for root, dirs, files in os.walk(base_path):
    for file in files:
        file_path = Path(root) / file
        file_type = '-'.join(file.split('-')[:3])
        file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')

        # Keep one instance of each type from today, delete others
        if file_date == today:
            if file_type not in file_types:
                file_types[file_type] = file_path
            else:
                os.remove(file_path)
        else:
            os.remove(file_path)