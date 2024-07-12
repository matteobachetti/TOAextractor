import os
import pandas as pd
import numpy as np
from astropy.table import Table
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Whisker, ColumnDataSource, PolyAnnotation, BoxAnnotation
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Category20b_20
from bokeh.layouts import column
from .utils.crab import retrieve_cgro_ephemeris
from astropy import units as u
from uncertainties import ufloat

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, "data")


def plot_frequency_history(fname, freq_units="mHz", output_fname=None, test=False):
    if output_fname is None:
        output_fname = "summary_freq.html"

    glitch_data = Table.read(os.path.join(datadir, "jb_crab_glitches.ecsv"))

    df = pd.read_csv(fname)
    res_label = "delta_f"
    df[res_label] = df["local_best_freq"] - df["initial_freq_estimate"]
    df[res_label + "_err"] = (
        np.abs(df["local_best_freq_err_n"]) + np.abs(df["local_best_freq_err_p"])
    ) / 2
    factor = ((1 * u.Hz) / (1 * u.Unit(freq_units))).to("").value
    for col in res_label, res_label + "_err":
        diff_str = col.replace(res_label, "")
        df[res_label + f"_{freq_units}" + diff_str] = df[col] * factor
    res_label = res_label + f"_{freq_units}"

    res_str = [
        f"{ufloat(res, err):P} {freq_units}"
        for res, err in zip(df[res_label], df[res_label + "_err"])
    ]
    df["freq_residual_str"] = res_str

    df["upper"] = np.array(df[res_label] + df[res_label + "_err"])
    df["lower"] = np.array(df[res_label] - df[res_label + "_err"])
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
                src="data:image/jpg;base64,@img" height="192" alt="Bla" width="248"
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
            <span style="font-size: 15px; color: #696;">@freq_residual_str</span>
        </div>
    </div>
    """
    p = figure(tooltips=TOOLTIPS, width=1200, height=800)

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
            y=res_label,
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
            line_alpha=0.1,
        )
        errorbar1.upper_head.size = 0
        errorbar1.lower_head.size = 0
        p.add_layout(errorbar1)
        errorbar2 = Whisker(
            base=res_label,
            upper="mjdstop",
            lower="mjdstart",
            source=source,
            dimension="width",
            level="annotation",
            line_width=2,
            line_color=color,
            line_alpha=0.1,
        )
        errorbar2.upper_head.line_color = color
        errorbar2.lower_head.line_color = color
        p.add_layout(errorbar2)
    p.title.text = "Residuals"
    p.xaxis.axis_label = "MJD"
    p.yaxis.axis_label = f"Residual ({freq_units})"
    p.legend.click_policy = "mute"

    return p


def plot_residuals(
    fname, time_units="us", res_label="fit_residual", output_fname=None, test=False
):
    eph_table = retrieve_cgro_ephemeris()
    if output_fname is None:
        output_fname = f"summary_{res_label}.html"

    glitch_data = Table.read(os.path.join(datadir, "jb_crab_glitches.ecsv"))
    output_file("TOAs.html")

    df = pd.read_csv(fname)

    factor = ((1 * u.s) / (1 * u.Unit(time_units))).to("").value
    for col in res_label, res_label + "_err":
        diff_str = col.replace(res_label, "")
        df[res_label + f"_{time_units}" + diff_str] = df[col] * factor
    res_label = res_label + f"_{time_units}"

    res_str = [
        f"{ufloat(res, err):P} {time_units}"
        for res, err in zip(df[res_label], df[res_label + "_err"])
    ]
    df["residual_str"] = res_str

    df["upper"] = np.array(df[res_label] + df[res_label + "_err"])
    df["lower"] = np.array(df[res_label] - df[res_label + "_err"])
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
                src="data:image/jpg;base64,@img" height="192" alt="Bla" width="248"
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
            <span style="font-size: 15px; color: #696;">@residual_str</span>
        </div>
    </div>
    """
    p = figure(tooltips=TOOLTIPS, width=1200, height=800)

    for row in eph_table:
        # The rms is in milliperiods, but here we use milliseconds.
        rms = row["RMS"] / 1000 / row["f0(s^-1)"] * factor

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
    poly = BoxAnnotation(
        fill_color="grey",
        top=(-0.000344 + 0.000040) * factor,  # Roths et al. 2004
        bottom=(-0.000344 - 0.000040) * factor,
        fill_alpha=0.2,
        line_width=2,
        line_alpha=0.2,
        line_color="black",
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
            y=res_label,
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
            line_alpha=0.1,
        )
        errorbar1.upper_head.size = 0
        errorbar1.lower_head.size = 0
        p.add_layout(errorbar1)
        errorbar2 = Whisker(
            base=res_label,
            upper="mjdstop",
            lower="mjdstart",
            source=source,
            dimension="width",
            level="annotation",
            line_width=2,
            line_color=color,
            line_alpha=0.1,
        )
        errorbar2.upper_head.line_color = color
        errorbar2.lower_head.line_color = color
        p.add_layout(errorbar2)
    p.title.text = "Residuals"
    p.xaxis.axis_label = "MJD"
    p.yaxis.axis_label = f"Residual ({time_units})"
    p.legend.click_policy = "mute"

    return p


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("file", help="Input summary CSV file", type=str)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "-o", "--output", help="Output file name", type=str, default="summary.html"
    )
    parser.add_argument(
        "-r",
        "--residual",
        help="Residual kind",
        choices=["fit", "toa"],
        default="fit",
    )
    parser.add_argument("--time-units", help="Time units", type=str, default="us")

    args = parser.parse_args(args)
    if args.residual == "toa":
        res_label = "residual"
    else:
        res_label = "fit_residual"

    p1 = plot_residuals(
        args.file,
        time_units=args.time_units,
        res_label=res_label,
        output_fname=args.output,
        test=args.test,
    )

    p2 = plot_frequency_history(
        args.file, freq_units="uHz", output_fname=None, test=False
    )
    p = column(p1, p2)
    output_file(args.output)
    save(p)
    show(p)
