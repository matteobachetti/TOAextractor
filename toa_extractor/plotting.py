import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("file", help="Input summary CSV file", type=str)

    args = parser.parse_args(args)

    output_file('TOAs.html')

    df = pd.read_csv(args.file)

    missions = list(set(df['mission']))
    p = figure()
    p.circle(x='mjd', y='residual',
             source=df,
             size=10, color=factor_cmap('mission', 'Category10_10', missions),
             legend='mission')
    p.title.text = 'Residuals'
    p.xaxis.axis_label = 'MJD'
    p.yaxis.axis_label = 'Residual (s)'

    hover = HoverTool()
    hover.tooltips = [
        ('Mission', '@mission'),
        ('MJD', '@mjd'),
        ('Instrument', '@instrument'),
        ('Residual', '@residual'),
    ]

    p.add_tools(hover)

    show(p)



