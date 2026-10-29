"""Check the distributions built for a release tag and choose where to publish.

Usage: python check_dist.py TAG DIST_DIR

DIST_DIR must hold the sdist and a wheel for every Python version and platform
built by .github/workflows/deploy.yml, all carrying the version of TAG (``v``
followed by a PEP 440 version). Prints ``testpypi`` for a pre-release tag
(``a``, ``b``, ``rc`` or ``dev``) and ``pypi`` for a final release.
"""
import pathlib
import sys

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version


# keep in line with [tool.cibuildwheel] in pyproject.toml
PYTHONS = ['cp312', 'cp313', 'cp314']
PLATFORMS = {
    'manylinux x86_64': lambda p: p.startswith('manylinux') and p.endswith('_x86_64'),
    'manylinux aarch64': lambda p: p.startswith('manylinux') and p.endswith('_aarch64'),
    'macOS arm64': lambda p: p.startswith('macosx') and p.endswith('_arm64'),
}


def check_dist(tag, dist_dir):
    if not tag.startswith('v'):
        sys.exit('release tag %r does not start with v' % tag)
    try:
        version = Version(tag[1:])
    except InvalidVersion:
        sys.exit('release tag %r is not v followed by a PEP 440 version' % tag)

    built = set()
    for f in sorted(pathlib.Path(dist_dir).iterdir()):
        if f.name.endswith('.whl'):
            _, file_version, _, tags = parse_wheel_filename(f.name)
            for t in tags:
                for platform, matches in PLATFORMS.items():
                    if matches(t.platform):
                        built.add((t.interpreter, platform))
        else:
            file_version = parse_sdist_filename(f.name)[1]
            built.add('sdist')
        if file_version != version:
            sys.exit('%s has version %s, but the tag is %s' % (f.name, file_version, tag))

    expected = {'sdist'} | {(py, p) for py in PYTHONS for p in PLATFORMS}
    missing = expected - built
    if missing:
        sys.exit('missing distributions: %s'
                 % ', '.join(sorted(m if m == 'sdist' else ' '.join(m) for m in missing)))

    return 'testpypi' if version.is_prerelease else 'pypi'


if __name__ == '__main__':
    print(check_dist(*sys.argv[1:]))
