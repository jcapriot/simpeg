import unittest
import os
import importlib
import glob

import matplotlib

matplotlib.use("Agg")


dirname, filename = os.path.split(os.path.abspath(__file__))
example_dir = dirname.split(os.path.sep)[:-2] + ["examples"]
dirs_to_test = ["07-nsem", "08-vrm", "09-flow", "20-published"]
print(example_dir, dirs_to_test)


class Doc_Test(unittest.TestCase):
    pass


def create_runner(script_path):
    def test_script(self):
        spec = importlib.util.spec_from_file_location("module.name", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        try:
            mod.run()  # some files are defined in a run command
        except:
            pass

    return test_script


# Programatically add tests to Examples
for dir in dirs_to_test:
    script_dir = os.path.sep.join(example_dir + [dir])
    os.chdir(script_dir)
    scripts = glob.glob(os.path.sep.join([script_dir] + ["*.py"]))
    scripts.sort()
    for script in scripts:
        script_name = "_".join(script.split(os.path.sep)[-2:])
        test_method = create_runner(script)
        test_method.__name__ = "test_" + script_name
        setattr(Doc_Test, test_method.__name__, test_method)
