import os
import glob
import pytest
from toa_extractor.pipeline import main, GetResidual, get_outputs


class TestPipeline(object):
    @classmethod
    def setup_class(cls):
        cls.curdir = os.path.dirname(__file__)
        cls.datadir = os.path.join(cls.curdir, "data")

    @pytest.mark.parametrize("mission", ["nustar", "astrosat"])
    def test_pipeline(self, mission):
        files = glob.glob(os.path.join(self.datadir, f"{mission}_test.*"))
        for f in files:
            for ext in [".evt", ".fits", ".ds"]:
                if ext not in f:
                    continue
                main([f])
                outputs = get_outputs(GetResidual(f, "none"))
                for outf in outputs:
                    assert os.path.exists(outf)
                    os.unlink(outf)

    @classmethod
    def teardown_class(cls):
        for ext in ["info", "template", "par"]:
            for product in glob.glob(os.path.join(cls.datadir, f"*.{ext}")):
                os.unlink(product)
