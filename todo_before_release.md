## Todos

- Run the tests.
  ```bash
  poetry run tox
  ```
- Check if docs are compilable:
  ```bash
  poetry run sphinx-build docs/source docs/_build
  ```
- Update release in `VERSION.md`, `setup.py` and button of `README.md`.
- Upload new packages to test server:
  ```bash
  poetry build
  poetry config repositories.testpypi https://test.pypi.org/legacy/
  poetry publish -r testpypi
  ```
- Check that the installation from the test server works, e.g. via 
  ```bash
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple innvestigate
  ```
- Tag commit according to [Semantic Versioning](https://semver.org) guidelines, e.g. `2.0.0`
- Go to releases tab on GitHub and "Create a new release"
- Upload new packages to real server:
  ```bash
  poetry publish
  ```
