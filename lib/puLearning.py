import pandas as pd
from pandas import DataFrame

from src import configuration
from lib import coTraining
from lib.classifier import Classifier


experimental_datasize = None
split_id_dir_path = None
pu_setup = None


class PuLearning:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, classifier: Classifier):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._classifier = classifier

    @property
    def classifier(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self.classifier

    @classifier.setter
    def classifier(self, classifier: Classifier):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._classifier = classifier

    def train(self, data: DataFrame, prop: str, target: str):
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        self.setup_data(data, prop, target)
        pass

    def setup(self, data: DataFrame):
        prop = "synth"
        self.setup_data(data, prop, prop)
        self._classifier.setup(data, prop, prop)

        configuration.config['General']['test_ratio']
        # from alignn_pu_config.py
        data_dir = configuration.input_dir
        root_dir = data_dir + "alignn_format"  # TODO set dynamically all alignn: maybe Classifier needs a name or maybe from project structure
        alignn_config_dir = "pu_alignn/alignn_configs"
        default_class_config = alignn_config_dir + "/default_class_config.json"
        class_config_name = alignn_config_dir + f'/class_config_{configuration.data_prefix}co{coTraining.co_step}alignn_{prop}.json'  # co{coTraining.co_step}alignn = {experiment}
        pu_config_name = alignn_config_dir + f'/pu_config_{configuration.data_prefix}co{coTraining.co_step}alignn_{prop}.json'  # co{coTraining.co_step}alignn = {experiment}

        # What should be configured in config file?
        global pu_setup
        pu_setup = dict()
        pu_setup["default_class_config"] = default_class_config
        pu_setup["pu_config_name"] = pu_config_name
        pu_setup["class_config_name"] = class_config_name
        pu_setup["data_dir"] = data_dir
        pu_setup["root_dir"] = root_dir
        pu_setup["file_format"] = "poscar"
        pu_setup["keep_data_order"] = False  # old comment: overwrites this attrib in config
        pu_setup["classification_threshold"] = 0.5  # old comment: also overwrites if present
        pu_setup["batch_size"] = None
        pu_setup["output_dir"] = None
        pu_setup["epochs"] = 120
        pu_setup["max_num_of_iterations"] = configuration.config['General']['number_of_iterations']
        pu_setup["start_of_iterations"] = configuration.config['General']['start_iteration']
        pu_setup["small_data"] = configuration.small_data

        # print(f'New PU Alignn pu_config_{data_prefix}{experiment}_{prop}.json was generated.')
        # not a file anymore but a global variable
        pass

    def setup_data(self, data: DataFrame, prop: str, TARGET: str):  # TODO how to give prop and target nice?
        # old pu_data_selection file: split data in Train and Test and save in {split_id_dir_path} directory

        # set path to directory for data for this step and make directory if does not exist
        global split_id_dir_path
        split_id_dir_path = configuration.project_path / f"{configuration.input_dir}/generated/{configuration.data_prefix}{TARGET}_{prop}"
        split_id_dir_path.mkdir(parents=True, exist_ok=True)

        # select data
        data = data[['material_id', prop, TARGET]]  # select
        data = data.loc[:, ~data.columns.duplicated()]  # drops duplicated props at round zero.
        data = data[~data[TARGET].isna()]  # removing NaN values. for small_data

        # TODO test how it looks
        # devide data in experimental data, leaveout data, positive data and save first and third seperately
        experimental_df = data[data[prop] == 1]  # dataframe with prop==1
        leaveoutdf = experimental_df.sample(frac=float(configuration.config['General']['test_ratio']) * 0.5,
                                            random_state=4242)  # TODO check if *0.5 is correct: old leaveout_test_portion set in file, frac see later
        positive_df = data[data[TARGET] == 1]  # labeled data
        positive_df = positive_df.drop(index=leaveoutdf.index)  # remove leave-out test data from positives

        # save ids for leave-out test data in file
        with open(split_id_dir_path / f"leaveout_test_id.txt", "w") as f:
            for test_id in leaveoutdf.index:
                f.write(str(test_id) + "\n")

        global experimental_datasize  # old: saved in a file
        experimental_datasize = experimental_df[prop].sum()

        # needed?
        # os.chdir(split_id_dir_path)  # ß new working directory

        for it in range(int(configuration.config['General']['num_iter'])):
            # select labeled test data: part positive labeled data and part labeled leaveout data
            testdf1 = positive_df.sample(frac=float(configuration.config['General']['test_ratio']), random_state=it) # select "test_ratio" positives, frac see later
            testdf1 = pd.concat([leaveoutdf, testdf1])  # add under leaveoutdf

            # full data without test data: devide in labeled train data and unlabeled data
            df_wo_test = data.drop(index=testdf1.index)  # remove test data
            traindf1 = df_wo_test[df_wo_test[TARGET] == 1].sample(frac=1, random_state=it + 1)  # train data (labeled), frac=1 means 100% of data
            unlabeled_df = df_wo_test[df_wo_test[TARGET] == 0]  # unlabeled data

            # difference between train data and unlabeled data
            unlabeled_shortage = len(traindf1) - len(unlabeled_df)

            if unlabeled_shortage > 0:
                testdf0 = unlabeled_df.sample(n=int(
                    configuration.config['General']['test_ratio'] * max(len(unlabeled_df), len(experimental_df))),
                                              random_state=it + 4)  # some unlabeled test data
                unlabeled_df = unlabeled_df.drop(index=testdf0.index)  # rest of unlabeled (without test data)
                # select unlabeled train data
                traindf0 = unlabeled_df.sample(frac=1,
                                               random_state=it + 2)  # a different 'negative' train-set at each iteration.
                traindf0_0 = unlabeled_df.sample(n=unlabeled_shortage, replace=True,
                                                 random_state=it + 3)
                traindf0 = pd.concat([traindf0,
                                      traindf0_0])  # Resampling is needed for co-training if more than half of all the data belongs to the positive class.
            else:
                traindf0 = unlabeled_df.sample(n=len(traindf1),
                                               random_state=it + 2)  # a different 'negative' train-set at each iteration.
                testdf0 = unlabeled_df.drop(index=traindf0.index)  # The remaining unlabeled data to be labeled.

            # join train data and save ids for it-th run
            it_traindf = pd.concat([traindf0, traindf1])
            it_traindf = it_traindf.sample(frac=1, random_state=it + 3)
            with open(split_id_dir_path / f"train_id_{it}.txt", "w") as f:
                for it_train_id in it_traindf.index:
                    f.write(str(it_train_id) + "\n")

            # join test data and save ids for it-th run
            it_testdf = pd.concat([testdf0, testdf1])  # positive test and unlabled prediction.
            it_testdf = it_testdf.sample(frac=1, random_state=it + 4)
            with open(split_id_dir_path / f"test_id_{it}.txt", "w") as f:
                for it_test_id in it_testdf.index:
                    f.write(str(it_test_id) + "\n")

# TODO setup and train func
# def functions for etc. data preparation, file save, file read, delete information from df
