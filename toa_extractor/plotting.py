import os
import pandas as pd
import numpy as np
from astropy.table import Table
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Whisker, ColumnDataSource, PolyAnnotation, BoxAnnotation
from bokeh.models import CDSView, GroupFilter

from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Category20b_20
from bokeh.layouts import column
from .utils.crab import retrieve_cgro_ephemeris
from .utils import root_name
from astropy import units as u
from uncertainties import ufloat

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, "data")


def get_data(fname, freq_units="mHz", time_units="us", res_label="fit_residual"):

    df = pd.read_csv(fname)

    # ---- Frequency residuals ----
    f_res_label = "delta_f"
    df[f_res_label] = df["local_best_freq"] - df["initial_freq_estimate"]
    df[f_res_label + "_err"] = (
        np.abs(df["local_best_freq_err_n"]) + np.abs(df["local_best_freq_err_p"])
    ) / 2
    factor = ((1 * u.Hz) / (1 * u.Unit(freq_units))).to("").value
    for col in f_res_label, f_res_label + "_err":
        diff_str = col.replace(f_res_label, "")
        df[f_res_label + f"_{freq_units}" + diff_str] = df[col] * factor

    f_res_label = f_res_label + f"_{freq_units}"

    res_str = [
        f"{ufloat(res, err):P} {freq_units}"
        for res, err in zip(df[f_res_label], df[f_res_label + "_err"])
    ]
    df["delta_f"] = df[f_res_label]
    df["delta_f_str"] = res_str

    df["delta_f_upper"] = np.array(df[f_res_label] + df[f_res_label + "_err"])
    df["delta_f_lower"] = np.array(df[f_res_label] - df[f_res_label + "_err"])

    # ---- Time residuals ----
    factor = ((1 * u.s) / (1 * u.Unit(time_units))).to("").value
    for col in res_label, res_label + "_err":
        diff_str = col.replace(res_label, "")
        df[res_label + f"_{time_units}" + diff_str] = df[col] * factor

    res_label = res_label + f"_{time_units}"

    res_str = [
        f"{ufloat(res, err):P} {time_units}"
        for res, err in zip(df[res_label], df[res_label + "_err"])
    ]
    df["delta_t"] = df[res_label]
    df["delta_t_str"] = res_str

    df["delta_t_upper"] = np.array(df[res_label] + df[res_label + "_err"])
    df["delta_t_lower"] = np.array(df[res_label] - df[res_label + "_err"])

    df["MJD_int"] = df["mjd"].astype(int)

    df["mission+instr"] = [
        f"{m}/{ins.upper()}" for m, ins in zip(df["mission"], df["instrument"])
    ]
    df["mission+ephem"] = [
        f"{m}-{e.upper()}" for m, e in zip(df["mission+instr"], df["ephem"])
    ]
    mission_ephem_combs = sorted(list(set(df["mission+ephem"])))
    mission_instr_combs = sorted(list(set(df["mission+instr"])))

    if len(mission_instr_combs) == len(mission_ephem_combs):
        mission_ephem_combs = mission_instr_combs
        df["mission+ephem"] = df["mission+instr"]

    if len(mission_instr_combs) == len(mission_ephem_combs):
        mission_ephem_combs = mission_instr_combs
        df["mission+ephem"] = df["mission+instr"]

    return ColumnDataSource(df)


def plot_frequency_history(
    full_dataset,
    glitch_data,
    res_label="Residuals",
    **figure_kwargs,
):

    df = full_dataset.to_df()
    mission_ephem_combs = sorted(list(set(df["mission+ephem"])))
    all_missions = sorted(list(set(df["mission"])))

    TOOLTIPS = """
    <div>
        <div>
            <img
                src="data:image/jpg;base64,@img" alt="Bla" width="248"
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
            <span style="font-size: 15px; color: #696;">@delta_f_str</span>
        </div>
    </div>
    """
    p = figure(tooltips=TOOLTIPS, **figure_kwargs)

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
        df_filt = df[df["mission+ephem"] == m]
        filt_source = ColumnDataSource(df_filt)
        group = GroupFilter(column_name="mission+ephem", group=m)
        view = CDSView(filter=group)

        p.scatter(
            x="mjd",
            y="delta_f",
            source=full_dataset,
            size=10,
            color=color,
            legend_label=m,
            muted_alpha=0.1,
            marker=markers,
            view=view,
        )
        errorbar1 = Whisker(
            base="mjd",
            upper="delta_f_upper",
            lower="delta_f_lower",
            source=filt_source,
            level="annotation",
            line_width=2,
            line_color=color,
            line_alpha=0.1,
        )
        errorbar1.upper_head.size = 0
        errorbar1.lower_head.size = 0
        p.add_layout(errorbar1)
        errorbar2 = Whisker(
            base="delta_f",
            upper="mjdstop",
            lower="mjdstart",
            source=filt_source,
            dimension="width",
            level="annotation",
            line_width=2,
            line_color=color,
            line_alpha=0.1,
        )
        errorbar2.upper_head.line_color = color
        errorbar2.lower_head.line_color = color
        p.add_layout(errorbar2)
    # p.title.text = "Residuals"
    p.xaxis.axis_label = "MJD"
    p.yaxis.axis_label = f"{res_label}"
    p.legend.click_policy = "mute"

    return p


def plot_residuals(
    full_dataset,
    glitch_data,
    res_label="Residuals",
    **figure_kwargs,
):
    # eph_table = retrieve_cgro_ephemeris()

    df = full_dataset.to_df()
    mission_ephem_combs = sorted(list(set(df["mission+ephem"])))
    all_missions = sorted(list(set(df["mission"])))

    TOOLTIPS = """
    <div>
        <div>
            <img
                src="data:image/jpg;base64,@img" alt="Bla" width="248"
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
            <span style="font-size: 15px; color: #696;">@delta_t_str</span>
        </div>
    </div>
    """
    p = figure(tooltips=TOOLTIPS, **figure_kwargs)

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
        group = GroupFilter(column_name="mission+ephem", group=m)
        # source = df
        view = CDSView(filter=group)

        df_filt = full_dataset.to_df()[df["mission+ephem"] == m]
        # source = ColumnDataSource(df_filt)
        filt_source = ColumnDataSource(df_filt)

        p.scatter(
            x="mjd",
            y="delta_t",
            source=full_dataset,
            size=10,
            color=color,
            legend_label=m,
            muted_alpha=0.1,
            marker=markers,
            view=view,
        )
        errorbar1 = Whisker(
            base="mjd",
            upper="delta_t_upper",
            lower="delta_t_lower",
            source=filt_source,
            level="annotation",
            line_width=2,
            line_color=color,
            line_alpha=0.1,
        )
        errorbar1.upper_head.size = 0
        errorbar1.lower_head.size = 0
        p.add_layout(errorbar1)
        errorbar2 = Whisker(
            base="delta_t",
            upper="mjdstop",
            lower="mjdstart",
            source=filt_source,
            dimension="width",
            level="annotation",
            line_width=2,
            line_color=color,
            line_alpha=0.1,
        )
        errorbar2.upper_head.line_color = color
        errorbar2.lower_head.line_color = color
        p.add_layout(errorbar2)
    # p.title.text = "Residuals"
    p.xaxis.axis_label = "MJD"
    p.yaxis.axis_label = f"{res_label}"
    p.legend.click_policy = "mute"

    return p


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("file", help="Input summary CSV file", type=str)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "-o", "--output", help="Output file name", type=str, default=None
    )
    parser.add_argument(
        "-r",
        "--residual",
        help="Residual kind",
        choices=["fit", "toa"],
        default="fit",
    )
    parser.add_argument("--time-units", help="Time units", type=str, default="us")
    parser.add_argument("--freq-units", help="Frequency units", type=str, default="uHz")

    args = parser.parse_args(args)
    if args.output is None:
        args.output = root_name(args.file) + ".html"

    if args.residual == "toa":
        res_label = "residual"
    else:
        res_label = "fit_residual"

    dataset = get_data(
        args.file,
        time_units=args.time_units,
        freq_units=args.freq_units,
        res_label=res_label,
    )

    glitch_data = Table.read(os.path.join(datadir, "jb_crab_glitches.ecsv"))

    time_units_str = u.Unit(args.time_units).to_string()
    freq_units_str = u.Unit(args.freq_units).to_string()
    if time_units_str.startswith("u"):
        time_units_str = r"\mu{}" + time_units_str[1:]
    if freq_units_str.startswith("u"):
        freq_units_str = r"\mu{}" + freq_units_str[1:]

    p1 = plot_residuals(
        dataset,
        glitch_data,
        width=1200,
        height=400,
        res_label=rf"$$\Delta{{\rm TOA}} ({time_units_str})$$",
    )

    eph_table = retrieve_cgro_ephemeris()
    time_factor = (1 * u.s / u.Unit(args.time_units)).to("").value
    for row in eph_table:
        # The rms is in milliperiods, but here we use milliseconds.
        rms = row["RMS"] / 1000 / row["f0(s^-1)"] * time_factor

        poly = PolyAnnotation(
            fill_alpha=0.1,
            fill_color="green",
            line_width=0,
            xs=[row["MJD1"], row["MJD1"], row["MJD2"], row["MJD2"]],
            ys=[-rms, rms, rms, -rms],
            level="image",
            propagate_hover=True,
        )
        p1.add_layout(poly)
    poly = BoxAnnotation(
        fill_color="grey",
        top=(-0.000344 + 0.000040) * time_factor,  # Roths et al. 2004
        bottom=(-0.000344 - 0.000040) * time_factor,
        fill_alpha=0.2,
        line_width=2,
        line_alpha=0.2,
        line_color="black",
        level="image",
        propagate_hover=True,
    )
    p1.add_layout(poly)
    p2 = plot_frequency_history(
        dataset,
        glitch_data,
        width=1200,
        height=400,
        res_label=rf"$$\Delta\nu_{{\rm spin}} {{ (\rm {freq_units_str}) }}$$",
    )
    p2.x_range = p1.x_range
    p = column(p1, p2)
    output_file(args.output)
    save(p)

    show(p)
