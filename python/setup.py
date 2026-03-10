import os
import subprocess
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel

NAME = "graph_compiler"
PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


def compile():
    env = os.environ
    # Do not use isolated env, if the project's virtual env is activated
    if "VIRTUAL_ENV" in env and os.path.dirname(
            env["VIRTUAL_ENV"]) == PROJ_DIR:
        if (py_path := env.get("PYTHONPATH", "")):
            print(f"Removing env.PYTHONPATH={py_path}", file=os.sys.stderr)
            env = env.copy()
            env.pop("PYTHONPATH")
    script_path = os.path.join(PROJ_DIR, "scripts", "compile.sh")
    print(f"Running build script: {script_path}", file=os.sys.stderr)
    subprocess.check_call(["/bin/sh", script_path], env=env)


class BuildPyCommand(build_py):

    def run(self):
        compile()
        super().run()


class EditableWheelCommand(editable_wheel):

    def run(self):
        compile()
        super().run()


class Dist(Distribution):

    def has_ext_modules(self):
        return True


setup(
    name=NAME,
    version="1.0.0",
    packages=[NAME],
    package_dir={NAME: "src"},
    package_data={NAME: ["*.so", "__init__.py"]},
    distclass=Dist,
    cmdclass={
        "build_py": BuildPyCommand,
        "editable_wheel": EditableWheelCommand,
    },
)
