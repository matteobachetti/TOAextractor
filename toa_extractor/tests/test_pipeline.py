import os
import glob
import pytest
from toa_extractor.pipeline import main


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
                if ext in f:
                    main([f])