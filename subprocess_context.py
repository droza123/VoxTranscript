import subprocess
import platform
import contextlib

DETACHED_PROCESS = 0x00000008

@contextlib.contextmanager
def silent_subprocess():
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def silent_popen(*args, **kwargs):
        if platform.system() == "Windows":
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | DETACHED_PROCESS
        return original_popen(*args, **kwargs)

    def silent_run(*args, **kwargs):
        if platform.system() == "Windows":
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | DETACHED_PROCESS
        return original_run(*args, **kwargs)

    try:
        subprocess.Popen = silent_popen
        subprocess.run = silent_run
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run