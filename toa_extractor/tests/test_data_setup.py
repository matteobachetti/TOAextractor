import glob
import os

import luigi
import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.table import Table
from stingray.gti import create_gti_mask

from toa_extractor.data_setup import GetPhaseogram

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, "data")


@pytest.fixture(scope="module")
def truncated_gti_file(tmp_path_factory):
    """NuSTAR test file whose GTI covers only the last 10% of the events."""
    outdir = tmp_path_factory.mktemp("truncated_gti")
    fname = str(outdir / "nustar_truncgti.evt")
    with fits.open(os.path.join(datadir, "nustar_test.evt.gz")) as hdul:
        times = hdul[1].data["TIME"]
        gti_hdu = hdul["GTI"]
        new_gti = np.array([[times[-1] - 0.1 * (times[-1] - times[0]), times[-1]]])
        gti_hdu.data = gti_hdu.data[:1]
        gti_hdu.data["START"] = new_gti[:, 0]
        gti_hdu.data["STOP"] = new_gti[:, 1]
        hdul.writeto(fname)
    n_in_gti = int(np.count_nonzero(create_gti_mask(times, new_gti)))
    return fname, new_gti, times.size, n_in_gti


def _run_phaseogram(fname, ignore_gtis):
    config_file = fname.replace(".evt", f"_ignore{ignore_gtis}.yaml")
    with open(config_file, "w") as fobj:
        yaml.dump({"format": "cgro", "ignore_gtis": ignore_gtis}, fobj)
    version = f"ignore{ignore_gtis}"
    assert luigi.build(
        [GetPhaseogram(fname, config_file, version)], local_scheduler=True, log_level="WARNING"
    )
    dynprofs = sorted(glob.glob(fname.replace(".evt", f"_{version}_dynprof_*.hdf5")))
    assert len(dynprofs) > 0
    return [Table.read(f) for f in dynprofs]


def test_phaseogram_truncated_gti(truncated_gti_file):
    """Events outside the GTIs must not be folded.

    Folding them used an extrapolated phase spline, which bent the phaseogram. (The warning
    about dropped events is emitted in a luigi child process, so it cannot be caught here.)
    """
    fname, gti, _, n_in_gti = truncated_gti_file
    tables = _run_phaseogram(fname, ignore_gtis=False)
    assert sum(np.sum(t["profile"]) for t in tables) == n_in_gti
    for t in tables:
        assert t.meta["time"][0] >= gti[0, 0]
