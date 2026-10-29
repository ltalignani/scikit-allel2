import pytest

from check_dist import check_dist


def touch(tmp_path, *names):
    for name in names:
        (tmp_path / name).touch()
    return tmp_path


def test_prerelease_tag_goes_to_testpypi(tmp_path):
    dist = touch(tmp_path,
                 'scikit_allel2-2.0.0.dev1.tar.gz',
                 'scikit_allel2-2.0.0.dev1-cp312-cp312-macosx_11_0_arm64.whl')
    assert check_dist('v2.0.0.dev1', dist) == 'testpypi'


@pytest.mark.parametrize('tag, version', [
    ('v2.0.0rc1', '2.0.0rc1'),
    ('v2.0.0b2', '2.0.0b2'),
    ('v2.0.0a1', '2.0.0a1'),
])
def test_other_prereleases_go_to_testpypi(tmp_path, tag, version):
    dist = touch(tmp_path, f'scikit_allel2-{version}.tar.gz')
    assert check_dist(tag, dist) == 'testpypi'


def test_final_tag_goes_to_pypi(tmp_path):
    dist = touch(tmp_path,
                 'scikit_allel2-2.0.0.tar.gz',
                 'scikit_allel2-2.0.0-cp314-cp314-manylinux_2_28_aarch64.whl')
    assert check_dist('v2.0.0', dist) == 'pypi'


def test_file_with_another_version_is_rejected(tmp_path):
    dist = touch(tmp_path,
                 'scikit_allel2-2.0.0rc1.tar.gz',
                 'scikit_allel2-1.3.14.dev3-cp312-cp312-macosx_11_0_arm64.whl')
    with pytest.raises(SystemExit, match='1.3.14.dev3'):
        check_dist('v2.0.0rc1', dist)


def test_empty_dist_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match='no distribution'):
        check_dist('v2.0.0', tmp_path)


def test_tag_without_v_is_rejected(tmp_path):
    dist = touch(tmp_path, 'scikit_allel2-2.0.0.tar.gz')
    with pytest.raises(SystemExit, match='tag'):
        check_dist('2.0.0', dist)
