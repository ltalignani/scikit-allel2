"""Check the distributions built for a release tag and choose where to publish.

Usage: python check_dist.py TAG DIST_DIR

Every wheel and sdist in DIST_DIR must carry the version of TAG (``v`` followed
by a PEP 440 version). Prints ``testpypi`` for a pre-release tag (``a``, ``b``,
``rc`` or ``dev``) and ``pypi`` for a final release.
"""
import pathlib
import sys

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version


def check_dist(tag, dist_dir):
    if not tag.startswith('v'):
        sys.exit('release tag %r does not start with v' % tag)
    try:
        version = Version(tag[1:])
    except InvalidVersion:
        sys.exit('release tag %r is not v followed by a PEP 440 version' % tag)

    files = sorted(pathlib.Path(dist_dir).iterdir())
    if not files:
        sys.exit('no distribution found in %s' % dist_dir)
    for f in files:
        if f.name.endswith('.whl'):
            file_version = parse_wheel_filename(f.name)[1]
        else:
            file_version = parse_sdist_filename(f.name)[1]
        if file_version != version:
            sys.exit('%s has version %s, but the tag is %s' % (f.name, file_version, tag))

    return 'testpypi' if version.is_prerelease else 'pypi'


if __name__ == '__main__':
    print(check_dist(*sys.argv[1:]))
