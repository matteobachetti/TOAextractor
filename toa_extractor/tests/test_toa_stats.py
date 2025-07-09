import os
from astropy.table import Table, vstack
import numpy as np


def create_fake_summary_file():
    data = {
        "mission": 20 * ["nustar"] + 3 * ["nicer"] + ["astrosat"],
        "instrument": 20 * ["fpma"] + 3 * ["xti"] + ["laxpc"],
        "obsid": [str(n) for n in np.random.randint(10000, 10000000, size=24)],
        "mjd": np.random.uniform(55000, 58000, 24),
        "ephem": ["DE200"] * 24,
        "fit_residual": np.random.normal(0, 1e-6, size=24),
        "fit_residual_err": np.random.chisquare(2, size=24) / 1e6,
    }
    print([len(d) for d in data.values()])
    df = Table(data)
    df = vstack([df, df])  # Duplicate the data to have more entries
    df["ephem"][24:] = "DE430"
    df["fit_residual"][24:] += 1e-6

    # Save to a CSV file
    summary_fname = "fake_summary.csv"
    df.write(summary_fname, overwrite=True)

    return summary_fname


def test_get_toa_stats():
    summary_fname = create_fake_summary_file()

    from toa_extractor.toa_stats import get_toa_stats

    outfname = "test_out_stats.csv"
    out_tex_fname = outfname.replace("csv", "tex")
    # Call the function to test
    table = get_toa_stats(
        summary_fname, out_fname=outfname, out_tex_fname=out_tex_fname
    )
    assert len(table) == 3
    assert "Mission" in table.colnames
    table.sort("Mission")  # Astrosat is the first line
    assert table["Mission"][0] == "ASTROSAT"
    assert table["Mission"][1] == "NICER"
    assert table["Mission"][2] == "NUSTAR"
    assert table["$N$"][0] == 1
    assert np.allclose(table[r"$\sigma_{\rm ephem}$ (us)"][1:], 0.5)

    assert np.isnan(table[r"$\sigma$ (us)"][0])

    # Check if the summary file was created and read correctly
    assert os.path.exists(outfname)
    assert os.path.exists(out_tex_fname)

    # Clean up
    os.unlink(outfname)
    os.unlink(out_tex_fname)
    os.unlink(summary_fname)
