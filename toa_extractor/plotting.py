import pandas as pd
from bokeh.plotting import figure, output_file, show, save
from bokeh.models.tools import HoverTool
from bokeh.transform import factor_cmap


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("file", help="Input summary CSV file", type=str)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "-o", "--output", help="Output file name", type=str, default="summary.html"
    )

    args = parser.parse_args(args)

    output_file("TOAs.html")

    df = pd.read_csv(args.file)

    missions = list(set(df["mission"]))
    p = figure()
    p.circle(
        x="mjd",
        y="residual",
        source=df,
        size=10,
        color=factor_cmap("mission", "Category10_10", missions),
        legend_label="mission",
    )
    p.title.text = "Residuals"
    p.xaxis.axis_label = "MJD"
    p.yaxis.axis_label = "Residual (s)"

    hover = HoverTool()
    hover.tooltips = [
        ("Mission", "@mission"),
        ("MJD", "@mjd"),
        ("Instrument", "@instrument"),
        ("Residual", "@residual"),
        ("ObsID", "@obsid"),
    ]

    p.add_tools(hover)

    output_file("summary.html")
    save(p)
    if not args.test:
        show(p)
