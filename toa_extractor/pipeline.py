import luigi
import yaml
from .utils.crab import get_crab_ephemeris
from .utils import root_name
from .utils.data_manipulation import get_observing_info


class GetResidual(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)
    def requires(self):
        return GetFoldedProfile(self.fname, self.config_file, self.worker_timeout)
    def output(self):
        pass
    def run(self):
        pass


class GetFoldedProfile(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)
    def requires(self):
        yield GetParfile(self.fname, self.config_file, self.worker_timeout)
        yield GetTemplate(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(root_name(self.fname) + "_folded.ecsv")

    def run(self):
        pass


class GetParfile(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)
    def requires(self):
        return GetInfo(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(root_name(self.fname) + ".par")
    def run(self):
        pass


class GetTemplate(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)
    def requires(self):
        return GetInfo(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(root_name(self.fname) + ".template")
    def run(self):
        pass


class GetInfo(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)

    def output(self):
        return luigi.LocalTarget(root_name(self.fname) + ".info")

    def run(self):
        info = get_observing_info(self.fname)
        with open(self.output.path, 'w') as f:
            yaml.dump(info, f)


def main(args=None):
    import argparse
    parser = \
        argparse.ArgumentParser(description="Automatic conversion from lv0 to "
                                            "lv1")

    parser.add_argument("files", help="Input binary files", type=str, nargs='+')
    parser.add_argument("--config", help="Config file", type=str,
                        default=None, required=True)
    parser.add_argument("--logfile",
                        help="Log file (default "
                             "{sta_id}_{idigit}_YYYY-MM-DD.log)", type=str,
                        default=None)
    parser.add_argument("--maxlevel", help="Maximum processing level", type=str,
                        default=None, choices=['LV0', 'LV0a', 'LV1', 'LV1a',
                                               'HM'])
    parser.add_argument("-f", "--force",
                        help="Force reprocessing of completed tasks",
                        action='store_true', default=False)
    parser.add_argument("--no-catch-log",
                        help="Do not catch all logs",
                        action='store_false', default=False)
    args = parser.parse_args(args)

    config_file = args.config

    for fname in args.files:
        res = luigi.build([GetInfo(fname, config_file)],
                          local_scheduler=True,
                          log_level='INFO',
                          workers=4)
