import subprocess
import sys
import pathlib
import platform
from multiprocessing import get_start_method
import pytest

# This test checks if the reinforcementLearningParallel.py script runs successfully with more than 1 workers.
# It is time capped at 600 seconds.

_is_linux_fork = platform.system() == "Linux" and get_start_method() == "fork"
_skip_non_linux_fork = pytest.mark.skipif(
    not _is_linux_fork,
    reason="ParallelEnv shared-memory cleanup segfaults on Windows (spawn). "
           "Test only runs on Linux with fork start method, which is the HPC deployment target.",
)
@_skip_non_linux_fork
def test_rlParallelRunsWith2Workers():
    # This test should pass on Linux with fork start method, and should not be run on Windows or MacOS.
    print("Is this test skipped ?")
    script = pathlib.Path(__file__).parent.parent / "src" / "qecc" / "reinforcementLearning.py"
    result = subprocess.run(
        [sys.executable, str(script), "--num-workers", "2"],
        timeout=600,
    )
    assert result.returncode == 0


if __name__ == "__main__":
    pass
