import glob
import os

import numpy as np
import pytest

from toa_extractor.pipeline import GetResidual, get_outputs
from toa_extractor.pipeline import main as main_pipe, select_n_files_per_directory
from toa_extractor.plotting import main as main_plotting
from toa_extractor.summary import main as main_summary
from toa_extractor.toa_stats import main as main_stats

version_label = f"test_{np.random.randint(0, 1000000)}"


class TestPipeline(object):
    @classmethod
    def setup_class(cls):
        cls.curdir = os.path.dirname(__file__)
        cls.datadir = os.path.join(cls.curdir, "data")

    @pytest.mark.parametrize("mission", ["nicer", "nustar", "astrosat"])
    def test_pipeline(self, mission):
        files = glob.glob(os.path.join(self.datadir, f"{mission}_test.*"))
        for f in files:
            for ext in [".evt", ".fits", ".ds"]:
                if ext not in f:
                    continue
                main_pipe([f, "--version", version_label, "--local-scheduler"])
                outputs = get_outputs(GetResidual(f, "none", version=version_label))
                for outf in outputs:
                    assert os.path.exists(outf)

        main_summary(
            glob.glob(os.path.join(self.datadir, f"*{version_label}*results*.yaml"))
            + ["--output", f"summary_{version_label}.csv"]
        )
        main_plotting([f"summary_{version_label}.csv", "--test"])
        main_stats([f"summary_{version_label}.csv"])

    @classmethod
    def teardown_class(cls):
        for product in glob.glob(os.path.join(cls.datadir, "*_test_test_[0-9]*")):
            os.unlink(product)


def test_select_n_files_lt_nmax():
    # Test selecting n files per directory
    # The complication in the path is needed for Windows compatibility
    input_files = [os.path.join(*name.split("/")) for name in ["./dir1/file1", "./dir2/file3"]]
    expected_output = [os.path.join(*name.split("/")) for name in ["dir1/file1", "dir2/file3"]]
    assert select_n_files_per_directory(input_files, 1) == expected_output


def test_select_n_files_gt_nmax():
    input_files = [
        os.path.join(*name.split("/"))
        for name in ["./dir1/file1", "./dir1/file2", "./dir2/file3", "./dir2/file4"]
    ]
    files = select_n_files_per_directory(input_files, 1)
    # Must be two files, from two different directories
    assert len(files) == 2
    assert len(set(os.path.split(f)[0] for f in files)) == 2
