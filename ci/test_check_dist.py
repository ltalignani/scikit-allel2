import pytest

from check_dist import check_dist


PLATFORMS = ['manylinux_2_17_x86_64.manylinux2014_x86_64',
             'manylinux_2_17_aarch64.manylinux2014_aarch64',
             'macosx_11_0_arm64']


def release(tmp_path, version, skip=()):
    """Create the files expected for `version`, except those named in `skip`."""
    names = ['scikit_allel2-%s.tar.gz' % version]
    for py in ('cp312', 'cp313', 'cp314'):
        for platform in PLATFORMS:
            names.append('scikit_allel2-%s-%s-%s-%s.whl' % (version, py, py, platform))
    for name in names:
        if name not in skip:
            (tmp_path / name).touch()
    return tmp_path


@pytest.mark.parametrize('tag', ['v2.0.0.dev1', 'v2.0.0rc1', 'v2.0.0b2', 'v2.0.0a1'])
def test_prerelease_tag_goes_to_testpypi(tmp_path, tag):
    assert check_dist(tag, release(tmp_path, tag[1:])) == 'testpypi'


def test_final_tag_goes_to_pypi(tmp_path):
    assert check_dist('v2.0.0', release(tmp_path, '2.0.0')) == 'pypi'


def test_file_with_another_version_is_rejected(tmp_path):
    dist = release(tmp_path, '2.0.0rc1')
    (dist / 'scikit_allel2-1.3.14.dev3-cp312-cp312-macosx_11_0_arm64.whl').touch()
    with pytest.raises(SystemExit, match='1.3.14.dev3'):
        check_dist('v2.0.0rc1', dist)


def test_missing_wheel_is_rejected(tmp_path):
    missing = 'scikit_allel2-2.0.0-cp313-cp313-manylinux_2_17_aarch64.manylinux2014_aarch64.whl'
    dist = release(tmp_path, '2.0.0', skip=[missing])
    with pytest.raises(SystemExit, match='cp313.*aarch64'):
        check_dist('v2.0.0', dist)


def test_missing_sdist_is_rejected(tmp_path):
    dist = release(tmp_path, '2.0.0', skip=['scikit_allel2-2.0.0.tar.gz'])
    with pytest.raises(SystemExit, match='sdist'):
        check_dist('v2.0.0', dist)


def test_tag_without_v_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match='tag'):
        check_dist('2.0.0', release(tmp_path, '2.0.0'))
