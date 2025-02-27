import multiprocessing as mp

class ProcessManager:
    """Manages multiple processes with shared synchronization events"""
    def __init__(self):
        self.processes = []
        self.shared_events = {
            'stopping_event': mp.Event(),
            'recording_event': mp.Event(),
        }
        
    def add_process(self, target, args=()):
        """Add a process with injected shared events"""
        # Inject shared events as keyword arguments
        final_args = args + (
            self.shared_events['stopping_event'],
            self.shared_events['recording_event'],
        )

        self.processes.append(
            mp.Process(target=target, args=final_args)
        )
        
    def start_all(self):
        """Start all registered processes"""
        for p in self.processes:
            p.start()
            
    def stop_all(self):
        """Gracefully stop all processes"""
        self.shared_events['stopping_event'].set()
        for p in self.processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()