import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Whisker, ColumnDataSource, PolyAnnotation
from bokeh.transform import factor_cmap
from .utils.crab import retrieve_cgro_ephemeris


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
    all_missions = sorted(list(set(df["mission+instr"])))

    if len(all_missions) == len(mission_ephem_combs):
        mission_ephem_combs = all_missions
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
            color=factor_cmap("mission+ephem", "Category20_20", mission_ephem_combs),
            legend_label=m,
            muted_alpha=0.1,
        )
        errorbar = Whisker(
            base="mjd",
            upper="upper",
            lower="lower",
            source=source,
            level="annotation",
            line_width=2,
            line_color=factor_cmap(
                "mission+ephem", "Category20_20", mission_ephem_combs
            ),
        )
        errorbar.upper_head.size = 0
        errorbar.lower_head.size = 0
        p.add_layout(errorbar)
    p.title.text = "Residuals"
    p.xaxis.axis_label = "MJD"
    p.yaxis.axis_label = "Residual (s)"
    p.legend.click_policy = "mute"
    # hover = HoverTool()
    # hover.tooltips = [
    #     ("Mission", "@mission"),
    #     ("MJD", "@mjd"),
    #     ("Instrument", "@instrument"),
    #     ("Residual", "@residual"),
    #     ("ObsID", "@obsid"),
    # ]

    # hover.tooltips = TOOLTIPS
    # p.add_tools(hover)

    output_file(args.output)
    save(p)
    if not args.test:
        show(p)
