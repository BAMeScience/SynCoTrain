import configparser
import pathlib

ehull015 = None
small_data = None
config = None
project_path = None
input_dir = None
result_dir = None
data_prefix = None


def configure(ehull, small):
    set_global_variables(ehull, small)
    global config
    config = read_config_file()


def read_config_file():  # Classifier input?
    """
    The Context delegates some work to the Strategy object instead of
    implementing multiple versions of the algorithm on its own.
    """
    path = pathlib.Path(__file__).parent.parent.absolute() / "config/config.ini"
    # Create a ConfigParser object
    c = configparser.ConfigParser()

    # Read the configuration file
    c.read(path)

    # Access values from the configuration file dynamically
    conf = {}
    for section in c.sections():
        section_config = {}
        for key in c.options(section):
            section_config[key] = (c.get(section, key))
        conf[section] = section_config

    # Return a dictionary with the retrieved values
    return conf


def set_global_variables(ehull, small):
    # set global variables
    global ehull015
    ehull015 = ehull
    global small_data
    small_data = small

    # check ehull015 and small_data configuration
    if ehull015 and small_data:
        error_message = "small_data and ehull015 are not allowed at the same time."
        raise Exception(error_message)

    # set global paths for project, input data and results
    global project_path
    project_path = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
    global input_dir
    input_dir = 'data/input'  # old variable propDFpath missing file
    global result_dir
    result_dir = 'data/results'

    # if paths dont exist, make directories
    (project_path / input_dir).mkdir(parents=True, exist_ok=True)
    (project_path / result_dir).mkdir(parents=True, exist_ok=True)

    # set global data_prefix
    global data_prefix
    data_prefix = "small_" if small_data else "15_" if ehull015 else ""


def current_setup(small_data, experiment, ehull015):
    if small_data:
        propDFpath = 'data/clean_data/small_synthDF'
        result_dir = 'data/results/small_data_synth'
        prop = 'synth'
    elif ehull015:
        propDFpath = 'data/clean_data/stabilityDF015'
        result_dir = 'data/results/stability015'
        prop = 'stability'
    else:
        propDFpath = 'data/clean_data/synthDF'
        result_dir = 'data/results/synth'
        prop = 'synth'

    experiment_target_match = {  # output_dir: training_label_column
        'alignn0': prop,
        'coAlignn1': 'schnet0',
        'coAlignn2': 'coSchnet1',
        'coAlignn3': 'coSchnet2',
        'coAlignn4': 'coSchnet3',
        'coAlignn5': 'coSchnet4',
        'schnet0': prop,
        'coSchnet1': 'alignn0',
        'coSchnet2': 'coAlignn1',
        'coSchnet3': 'coAlignn2',
        'coSchnet4': 'coAlignn3',
        'coSchnet5': 'coAlignn4',
        'final_avg': 'final_label',
    }

    return {"propDFpath": propDFpath, "result_dir": result_dir, "prop": prop,
            "TARGET": experiment_target_match[experiment]}
# TODO current_setup function from experiment_setup.py
