import os
import glob
import pytest
from toa_extractor.pipeline import GetResidual, get_outputs
from toa_extractor.pipeline import main as main_pipe
from toa_extractor.summary import main as main_summary
from toa_extractor.plotting import main as main_plotting


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
                main_pipe([f])
                outputs = get_outputs(GetResidual(f, "none"))
                for outf in outputs:
                    assert os.path.exists(outf)

        main_summary(glob.glob(os.path.join(self.datadir, "*residual.yaml")))
        main_plotting(["summary.csv", "--test"])

    @classmethod
    def teardown_class(cls):

        for product in glob.glob(os.path.join(cls.datadir, "*_none*")):
            os.unlink(product)
