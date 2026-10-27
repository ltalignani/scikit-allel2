# -*- coding: utf-8 -*-
import pytest


@pytest.fixture(autouse=True)
def _doctest_in_tmp_path(request, tmp_path, monkeypatch):
    # doctests write example files (callset.h5, example.vcf...): keep them out of the
    # working directory, and apart from each other
    if isinstance(request.node, pytest.DoctestItem):
        monkeypatch.chdir(tmp_path)
