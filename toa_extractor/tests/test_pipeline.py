import glob
import os

import numpy as np
import pytest

from toa_extractor.pipeline import GetResidual, get_outputs
from toa_extractor.pipeline import main as main_pipe
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
                main_pipe([f, "--version", version_label])
                outputs = get_outputs(GetResidual(f, "none", version=version_label))
                print(outputs)
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
