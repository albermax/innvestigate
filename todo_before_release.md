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
- Make release
- Push master and develop branch, and tags
- Switch to master branch
- Upload new packages to test server:
  ```bash
  poetry build
  poetry config repositories.testpypi https://test.pypi.org/legacy/
  poetry publish -r testpypi
  ```
- Check that everything works
- Upload new packages to real server:
  ```bash
  poetry publish
  ```
