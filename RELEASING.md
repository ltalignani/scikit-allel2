# Releasing scikit-allel2

The version comes from the git tag (setuptools-scm). Pushing a tag `v*` runs
`.github/workflows/deploy.yml`: it builds the wheels (CPython 3.12–3.14,
manylinux x86_64 and aarch64, macOS arm64) and the sdist, smoke-tests every
wheel, then waits for approval before publishing:

| Tag | Example | Published to |
|---|---|---|
| pre-release (`a`, `b`, `rc`, `dev`) | `v2.0.0rc1` | TestPyPI |
| final release | `v2.0.0` | PyPI |

1. **Release notes.** On `master`, the section for the version in
   `docs/release.rst` is complete and merged.

2. **Tag.**

   ```bash
   git switch master && git pull
   git tag -a v2.0.0rc1 -m v2.0.0rc1
   git push origin v2.0.0rc1
   ```

3. **Approve.** In the Actions tab, open the *Deploy* run for the tag. Once the
   builds pass, *Review deployments* and approve the `testpypi` or `pypi`
   environment.

4. **Check the package.** In a clean environment, on Linux and macOS:

   ```bash
   # TestPyPI does not host the dependencies, hence the extra index
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ scikit-allel2==2.0.0rc1
   # PyPI
   pip install scikit-allel2==2.0.0
   python -c "import allel2; print(allel2.__version__)"
   ```

5. **Check the documentation.** ReadTheDocs builds the tag; after a final
   release, https://scikit-allel2.readthedocs.io/en/stable/ shows the new
   version.

A version cannot be uploaded twice to the same index: if a publication fails
after the upload, fix the problem and tag the next version (`rc2`, `dev2`).
