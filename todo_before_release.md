## To-do-list for package releases
- Pass test suite:
  ```bash
  poetry run tox
  ```
- Increase the version number in the `project.toml` and `/src/innvestigate/__init__.py`
- Write release notes in `VERSION.md`
- Check if docs are compilable:
  ```bash
  poetry run sphinx-build docs/source docs/_build
  ```
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
- Tag commit according to [Semantic Versioning](https://semver.org) guidelines
- Go to releases tab on GitHub and "Create a new release", manually including the `.tar.gz` and `.whl` files from the build
- Upload new packages to PyPI:
  ```bash
  poetry publish
  ```
