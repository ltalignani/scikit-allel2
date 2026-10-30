# Releasing scikit-allel2

The version comes from the git tag (setuptools-scm). Pushing a tag `v*` runs
`.github/workflows/deploy.yml`: it builds the wheels (CPython 3.12–3.14,
manylinux x86_64 and aarch64, macOS arm64) and the sdist, smoke-tests every
wheel, then waits for approval before publishing:

| Tag | Example | Published to |
|---|---|---|
| pre-release (`a`, `b`, `rc`) | `v2.0.0rc1` | TestPyPI |
| final release | `v2.0.0` | PyPI |

1. **Release notes.** On `master`, the section for the version in
   `docs/release.rst` is complete and merged. For a final release, the
   classifier in `pyproject.toml` is `Development Status :: 5 - Production/Stable`
   (`4 - Beta` for pre-releases).

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

5. **Release candidates: publish to PyPI.** Once a release candidate checks
   out on TestPyPI, publish the same files to PyPI, so that users can test it
   with `pip install --pre scikit-allel2`: in the Actions tab, *Deploy*,
   *Run workflow*, choose the tag (`v2.0.0rc1`, not a branch) under *Use
   workflow from*, run, then approve the `pypi` environment. The workflow
   downloads the files from TestPyPI and checks their hashes and version; it
   does not rebuild them.

6. **Check the documentation.** ReadTheDocs builds the tag. Its `stable`
   version follows the highest final release tag: after the first final
   release, reactivate `stable` (ReadTheDocs, *Versions*), then check that
   https://scikit-allel2.readthedocs.io/en/stable/ shows the new version.

A version cannot be uploaded twice to the same index: if a publication fails
after the upload, fix the problem and tag the next version (`rc2`).

Use `a`, `b` or `rc` tags for pre-releases, not `.devN`: setuptools-scm cannot
compute the version of the commits after a `.devN` tag (N > 0), which breaks
every build on `master` until the tag is deleted.
