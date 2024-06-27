import os
import pandas as pd
import numpy as np
from astropy.table import Table
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Whisker, ColumnDataSource, PolyAnnotation, BoxAnnotation
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Category20b_20
from .utils.crab import retrieve_cgro_ephemeris

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, "data")


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("file", help="Input summary CSV file", type=str)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "-o", "--output", help="Output file name", type=str, default="summary.html"
    )

    args = parser.parse_args(args)

    eph_table = retrieve_cgro_ephemeris()
    glitch_data = Table.read(os.path.join(datadir, "jb_crab_glitches.ecsv"))
    output_file("TOAs.html")

    df = pd.read_csv(args.file)
    df["upper"] = np.array(df["residual"] + df["residual_err"])
    df["lower"] = np.array(df["residual"] - df["residual_err"])
    df["MJD_int"] = df["mjd"].astype(int)

    df["mission+instr"] = [
        f"{m}/{ins.upper()}" for m, ins in zip(df["mission"], df["instrument"])
    ]
    df["mission+ephem"] = [
        f"{m}-{e.upper()}" for m, e in zip(df["mission+instr"], df["ephem"])
    ]
    mission_ephem_combs = sorted(list(set(df["mission+ephem"])))
    mission_instr_combs = sorted(list(set(df["mission+instr"])))
    all_missions = sorted(list(set(df["mission"])))

    if len(mission_instr_combs) == len(mission_ephem_combs):
        mission_ephem_combs = mission_instr_combs
        df["mission+ephem"] = df["mission+instr"]

    TOOLTIPS = """
    <div>
        <div>
            <img
                src="data:image/jpg;base64,@img" height="96" alt="Bla" width="128"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@mission</span>
            <span style="font-size: 15px; color: #966;">[@instrument]</span>
        </div>
        <div>
            <span>ObsID @obsid</span>
        </div>
        <div>
            <span style="font-size: 15px; color: #696;">@residual</span>
        </div>
    </div>
    """
    p = figure(tooltips=TOOLTIPS, width=1200, height=800)

    for row in eph_table:
        rms = row["RMS"] / 1000 / row["f0(s^-1)"]

        poly = PolyAnnotation(
            fill_alpha=0.1,
            fill_color="green",
            line_width=0,
            xs=[row["MJD1"], row["MJD1"], row["MJD2"], row["MJD2"]],
            ys=[-rms, rms, rms, -rms],
        )
        p.add_layout(poly)

    for row in glitch_data:
        mjd = float(row["MJD"])
        mjde = float(row["MJDe"])

        poly = BoxAnnotation(
            fill_color="red",
            left=mjd - 0.5 * mjde,
            right=mjd + 0.5 * mjde,
            fill_alpha=0.2,
            line_width=2,
            line_alpha=0.2,
            line_color="red",
        )
        p.add_layout(poly)

    color = factor_cmap(
        "mission+ephem",
        palette=Category20b_20,
        factors=mission_ephem_combs,
        end=len(mission_ephem_combs),
    )
    MARKERS = [
        "asterisk",
        "circle",
        "diamond",
        "hex",
        "inverted_triangle",
        "square",
        "star",
        "triangle",
        "triangle_dot",
        "star_dot",
        "square_cross",
        "hex_dot",
        "plus",
        "square_dot",
        "square_pin",
        "square_x",
        "triangle_pin",
        "x",
        "y",
        "circle_cross",
        "circle_dot",
        "circle_x",
        "circle_y",
        "cross",
        "dash",
        "diamond_cross",
        "diamond_dot",
        "dot",
    ]

    markers = factor_mark("mission", MARKERS, factors=all_missions)
    for m in mission_ephem_combs:
        print(m)
        df_filt = df[df["mission+ephem"] == m]
        source = ColumnDataSource(df_filt)

        print(df_filt)
        p.scatter(
            x="mjd",
            y="residual",
            source=source,
            size=10,
            color=color,
            legend_label=m,
            muted_alpha=0.1,
            marker=markers,
        )
        errorbar1 = Whisker(
            base="mjd",
            upper="upper",
            lower="lower",
            source=source,
            level="annotation",
            line_width=2,
            line_color=color,
        )
        errorbar1.upper_head.size = 0
        errorbar1.lower_head.size = 0
        p.add_layout(errorbar1)
        errorbar2 = Whisker(
            base="residual",
            upper="mjdstop",
            lower="mjdstart",
            source=source,
            dimension="width",
            level="annotation",
            line_width=2,
            line_color=color,
        )
        errorbar2.upper_head.line_color = color
        errorbar2.lower_head.line_color = color
        p.add_layout(errorbar2)
    p.title.text = "Residuals"
    p.xaxis.axis_label = "MJD"
    p.yaxis.axis_label = "Residual (s)"
    p.legend.click_policy = "mute"

    output_file(args.output)
    save(p)
    if not args.test:
        show(p)
