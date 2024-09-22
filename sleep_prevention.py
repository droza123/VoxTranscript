import platform
import subprocess
import ctypes
import atexit

class SleepPrevention:
    def __init__(self):
        self.system = platform.system()
        self.prevention_active = False

    def prevent_sleep(self):
        if self.prevention_active:
            return

        if self.system == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
        elif self.system == "Darwin":  # macOS
            subprocess.Popen(["caffeinate", "-i", "-w", str(subprocess.Popen.pid)])
        elif self.system == "Linux":
            subprocess.Popen(["systemd-inhibit", "--what=sleep", "--why=Transcription in progress", "sleep", "infinity"])

        self.prevention_active = True
        atexit.register(self.allow_sleep)

    def allow_sleep(self):
        if not self.prevention_active:
            return

        if self.system == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        elif self.system == "Darwin":  # macOS
            subprocess.run(["killall", "caffeinate"])
        elif self.system == "Linux":
            subprocess.run(["killall", "systemd-inhibit"])

        self.prevention_active = False
        atexit.unregister(self.allow_sleep)

sleep_preventer = SleepPrevention()