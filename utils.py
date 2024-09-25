import os
import logging
import sys
import appdirs
import tempfile
import torch
import gc
import pynvml

APP_NAME = "VoxTranscript"
APP_AUTHOR = None

def get_app_dir(dir_type):
    """
    Get the appropriate application directory based on the type.
    :param dir_type: String, either 'data', 'config', 'log', or 'temp'
    :return: String, the full path to the requested directory
    """
    if dir_type == 'data':
        return appdirs.user_data_dir(APP_NAME, APP_AUTHOR)
    elif dir_type == 'config':
        return appdirs.user_config_dir(APP_NAME, APP_AUTHOR)
    elif dir_type == 'log':
        return appdirs.user_log_dir(APP_NAME, APP_AUTHOR)
    elif dir_type == 'temp':
        return tempfile.gettempdir()
    else:
        raise ValueError(f"Unknown directory type: {dir_type}")

def ensure_dir_exists(dir_path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(dir_path, exist_ok=True)

def get_app_file_path(file_name, dir_type):
    """
    Get the full path for an application file.
    :param file_name: String, the name of the file
    :param dir_type: String, either 'data', 'config', 'log', or 'temp'
    :return: String, the full path to the file
    """
    dir_path = get_app_dir(dir_type)
    ensure_dir_exists(dir_path)
    return os.path.join(dir_path, file_name)

def check_gpu_availability():
    if not torch.cuda.is_available():
        return False

    cuda_version = torch.version.cuda
    if cuda_version:
        major, minor = map(int, cuda_version.split('.')[:2])
        if (major, minor) < (11, 8):  # Minimum required version
            return False

    cuda_capability = torch.cuda.get_device_capability(0)
    required_capability = (6, 1)  # Minimum requirement for CUDA 11.8
    if cuda_capability < required_capability:
        return False

    return True

def force_cuda_memory_release():
    if torch.cuda.is_available():
        # Clear PyTorch's CUDA memory cache
        torch.cuda.empty_cache()
        
        # Force CUDA to release all memory
        torch.cuda.synchronize()
        
        # Attempt to release all unoccupied cached memory
        torch.cuda.memory_allocated()
        torch.cuda.memory_cached()
        
        # Run garbage collection
        gc.collect()
        
        logging.info("Forced CUDA memory release")
    else:
        logging.info("CUDA not available, skipping memory release")
        
def log_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        logging.info(f"GPU Memory: Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")
        for i in range(torch.cuda.device_count()):
            usage, cache, misc = gpu_memory_usage_all(i)
            logging.info(f"GPU {i} Memory:")
            logging.info(f"  Used: {usage:.3f} GB")
            logging.info(f"  Cached: {cache:.3f} GB")
            logging.info(f"  Misc: {misc:.3f} GB")
    else:
        logging.info("CUDA not available, cannot log GPU memory usage")

def gpu_memory_usage_all(device=0):
    usage = torch.cuda.memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.memory_reserved(device) / 1024.0**3
    smi = gpu_memory_usage_smi(device)
    return usage, reserved - usage, max(0, smi - reserved)

def gpu_memory_usage_smi(device=0):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024.0**3

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running in normal Python environment
        base_path = os.path.dirname(os.path.dirname(__file__))
    
    full_path = os.path.normpath(os.path.join(base_path, relative_path))
    
    # If the file doesn't exist and we're in development, try the root directory
    if not os.path.exists(full_path) and not getattr(sys, 'frozen', False):
        full_path = os.path.normpath(os.path.join(os.path.dirname(__file__), relative_path))
    
    return full_path

    # Example usage:
    # icon_path = resource_path(os.path.join("icons", "voxtranscript-icon.svg"))

    # When creating an icon with a variable name:
    # icon_name = "some-icon-name"
    # icon = QIcon(resource_path(os.path.join("icons", f"{icon_name}.svg")))
    # Then, when you need to reference an icon:
    # icon_path = resource_path(os.path.join("icons", "your_icon_name.svg"))