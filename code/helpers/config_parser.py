import os
import json


def str2bool(v):
    """
    Function that return a boolean from a string.

    Parameters
    ----------
    v: str
        input string

    Returns
    -------
    bool
    """
    return v.lower() in ("yes", "true", "t", "1")


class JSONRunConfig:
    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            params = json.load(f)

        self.model = params.get('model', 'lodnn')
        self.dataset = params.get('dataset', 'kitti')
        self.view = params.get('view', 'bev')
        self.parameters = params.get('parameters', None)
        self.shape = tuple(params.get('shape', (400, 200)))
        self.output_channels = params.get('output_channels', 2)
        sequences = params.get('sequences', None)

        if sequences is not None:
            self.sequences = ['{:02}'.format(i) for i in sequences]
        else:
            self.sequences = None

        self.gt_args = params.get('gt_args', None)
        self.store_features_basedir = params.get('store_features_basedir', None)
        self.training_config = params.get('training_config')
        self.experiments = params.get('experiments', None)

        self.features = dict(
            compute_classic=str2bool(self.parameters.get('compute_classic', 0)),
            add_geometrical_features=str2bool(self.parameters.get('add_geometrical_features', 0)),
            subsample_ratio=self.parameters.get('subsample_ratio', 1),
            compute_eigen=self.parameters.get('compute_eigen', 0),
            compute_z=False,
        )

        if self.store_features_basedir is not None:
            layer_dir = str(64 // self.features['subsample_ratio'])
            self.features_basedir = os.path.join(self.store_features_basedir, self.dataset, self.view, layer_dir)
        else:
            self.features_basedir = ''

        self.config_run = dict(model=self.model,
                               shape=self.shape,
                               output_channels=self.output_channels,
                               dataset=self.dataset,
                               view=self.view,
                               sequences=self.sequences,
                               features_basedir=self.features_basedir,
                               training_config=self.training_config,
                               experiments=self.experiments,
                               features=self.features,
                               gt_args=self.gt_args)

    def add_str_attr(self, attr):
        """
        Parameters
        ----------
        attr: str

        Returns
        -------
        str
        """
        if hasattr(self, attr):
            return attr + ": " + getattr(self, attr).__str__() + "\n"

    def __str__(self):
        out = ""
        out += self.add_str_attr("model")
        out += self.add_str_attr("shape")
        out += self.add_str_attr("output_channels")
        out += self.add_str_attr("dataset")
        out += self.add_str_attr("view")
        out += self.add_str_attr("parameters")
        out += self.add_str_attr("sequences")
        out += self.add_str_attr("gt_args")
        out += self.add_str_attr("store_features_basedir")
        out += self.add_str_attr("training_config")
        out += self.add_str_attr("experiments")
        out += self.add_str_attr("features")
        out += self.add_str_attr("features_basedir")
        out += self.add_str_attr("config_run")

        return out

    def get_config_run(self):
        return self.config_run

    def update_config_run(self, experiment_name):

        base_feat = self.features
        features = dict(
            compute_classic=False,
            add_geometrical_features=False,
            subsample_ratio=base_feat.get('subsample_ratio', 1),
            compute_eigen=0,
            compute_z=False,
        )

        if 'classic' in experiment_name.lower():
            features['compute_classic'] = True

        if 'geometry' in experiment_name.lower():
            features['add_geometrical_features'] = True

        if 'eigen' in experiment_name.lower():
            features['compute_eigen'] = base_feat.get('compute_eigen')

        if 'height' in experiment_name.lower():
            features['compute_z'] = True

        self.config_run['features'] = features

    def get_test_name(self):
        """
        Function that return test name based on the features parameters

        Return
        ------
        test_name: str
            name of the test file

        """
        features = self.config_run.get('features')
        test_name = 'Classical' if features['compute_classic'] else ''
        test_name += '_Geometric' if features['add_geometrical_features'] else ''
        test_name += '_Eigen' if features['compute_eigen'] else ''

        if features['subsample_ratio'] == 2:
            test_name += '_Subsampled_32'

        if features['subsample_ratio'] == 4:
            test_name += '_Subsampled_16'

        return test_name