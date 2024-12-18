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
    path = pathlib.Path(__file__).parent.parent.parent.absolute() / "config.ini"
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
    project_path = pathlib.Path(__file__).parent.parent.parent.absolute()
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
