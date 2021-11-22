"""
This is a small utility to invoke MonkeyType by calling

    poetry run monkeytype run tests/_monkeytype.py

It will run all tests and automatically infer type annotations.
Modules with stubs can be listed by

    poetry run monkeytype list-modules

The generated stub can be printed using

    poetry run monkeytype stub innvestigate.foo.bar

and applied with

    poetry run monkeytype apply innvestigate.foo.bar

!!! NOTE !!!
Automated type annotations __absolutely__ need to be checked using mypy
as well as formated using black and isort before commiting.
"""
import pytest

pytest.main()
