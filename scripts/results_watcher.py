import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ResultsRefreshHandler(FileSystemEventHandler):
    def __init__(self, refresh_callback):
        self.refresh_callback = refresh_callback

    def on_modified(self, event):
        if event.src_path.endswith(".json"):
            self.refresh_callback()

def start_results_watcher(directory, refresh_callback):
    event_handler = ResultsRefreshHandler(refresh_callback)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Example usage
def refresh_results():
    print("Results have been updated. Refreshing...")

if __name__ == "__main__":
    results_directory = "artifacts/comparisons"
    start_results_watcher(results_directory, refresh_results)