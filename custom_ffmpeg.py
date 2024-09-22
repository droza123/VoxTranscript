import logging
from ffmpeg.nodes import output_operator
from ffmpeg._run import Error, compile
import subprocess
import platform

DETACHED_PROCESS = 0x00000008

def run_subprocess(*args, **kwargs):
    if platform.system() == "Windows":
        kwargs['creationflags'] = DETACHED_PROCESS
    return subprocess.run(*args, **kwargs)

def popen_subprocess(*args, **kwargs):
    if platform.system() == "Windows":
        kwargs['creationflags'] = DETACHED_PROCESS
    return subprocess.Popen(*args, **kwargs)

@output_operator()
def run_async(
    stream_spec,
    cmd='ffmpeg',
    pipe_stdin=False,
    pipe_stdout=False,
    pipe_stderr=False,
    quiet=False,
    overwrite_output=False,
):
    DETACHED_PROCESS = 0x00000008
    args = compile(stream_spec, cmd, overwrite_output=overwrite_output,)
    logging.debug(f"FFMPEG command: {' '.join(args)}")
    stdin_stream = subprocess.PIPE if pipe_stdin else None
    stdout_stream = subprocess.PIPE if pipe_stdout or quiet else None
    stderr_stream = subprocess.PIPE if pipe_stderr or quiet else None
    return subprocess.Popen(args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream, creationflags=DETACHED_PROCESS)

@output_operator()
def custom_ffmpeg_run(
    stream_spec,
    cmd='ffmpeg',
    capture_stdout=False,
    capture_stderr=False,
    input=None,
    quiet=False,
    overwrite_output=False,
):
    process = run_async(
        stream_spec,
        cmd,
        pipe_stdin=input is not None,
        pipe_stdout=capture_stdout,
        pipe_stderr=capture_stderr,
        quiet=quiet,
        overwrite_output=overwrite_output,  # Pass this parameter
    )
    out, err = process.communicate(input)
    retcode = process.poll()
    if retcode:
        logging.error(f"FFMPEG process returned non-zero exit code: {retcode}")
        logging.info(f"FFMPEG stderr: {err.decode() if err else 'None'}")
        raise Error('ffmpeg', out, err)
    return out, err