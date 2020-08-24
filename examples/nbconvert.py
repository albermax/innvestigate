# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import argparse
import os
import sys
import subprocess


###############################################################################
###############################################################################
###############################################################################


if __name__ == "__main__":
    def is_executable(filepath):
        #determine whether the target file exists and is executable
        return os.path.isfile(filepath) and os.access(filepath, os.X_OK)

    # Try jupyter binary that is the same directory as the running python.
    # E.g., useful for python venvs.
    bin_dir = os.path.dirname(sys.executable)
    jupyter_executable = os.path.join(bin_dir, "jupyter")

    if not is_executable(jupyter_executable):
        # Fallback to find any jupyter (and hope for the best).
        jupyter_executable = subprocess.check_output(["which jupyter"],
                                                     shell=True).strip()

    #assert a valid executable of jupyter has been found
    assert is_executable(jupyter_executable), 'No executable for "jupyter" could be found'



    # Get all notebooks
    notebook_dir = os.path.join(os.path.dirname(__file__), "notebooks")
    notebooks = [x for x in os.listdir(notebook_dir) if x.endswith(".ipynb")]

    output_dir = os.path.join(os.path.dirname(__file__), "nbconvert_tmp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = argparse.ArgumentParser(
        description="Script to handle the example notebooks via command line.")
    parser.add_argument(
        'command', choices=["execute", "to_script", "to_script_and_execute"],
        help="What to do.")
    args = parser.parse_args()

    if args.command == "execute":
        for notebook in notebooks:
            print("=" * 80)
            print("=" * 80)
            print("=" * 80)
            print("Running notebook:", notebook)
            print("-" * 80)
            print()

            call = [
                jupyter_executable, "nbconvert",
                "--output-dir='%s'" % output_dir,
                "--ExecutePreprocessor.timeout=-1",
                "--to", "notebook", "--execute",
                os.path.join(notebook_dir, notebook)
            ]
            subprocess.check_call(call)
    elif args.command in ["to_script", "to_script_and_execute"]:
        for notebook in notebooks:
            print("Convert notebook:", notebook)
            input_file = os.path.join(notebook_dir, notebook)
            output_file = os.path.join(
                output_dir, notebook.replace(".ipynb", ".py"))

            call = [
                jupyter_executable, "nbconvert",
                "--output-dir='%s'" % output_dir,
                "--to", "script",
                input_file,
            ]
            subprocess.check_call(call)

            # Fix magic lines
            with open(output_file, "r", encoding="utf-8") as f:
                script_content = f.read()

            fixed_lines = []
            for line in script_content.split("\n"):
                if "get_ipython()" in line:
                    line = "# %s" % line
                fixed_lines.append(line)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(fixed_lines))

            if args.command == "to_script_and_execute":
                subprocess.check_call(
                    [sys.executable, notebook.replace(".ipynb", ".py")],
                    cwd=output_dir)
    else:
        raise ValueError("Command not recognized")
