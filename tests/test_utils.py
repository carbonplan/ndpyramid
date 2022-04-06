import pytest

import ndpyramid


@pytest.mark.parametrize(
    'datasets,type,method,version,args,kwargs', [(None, '', '', '', None, None)]
)
def test_multiscales_template(datasets, type, method, version, args, kwargs):
    template = ndpyramid.utils.multiscales_template(
        datasets=datasets, type=type, method=method, version=version, args=args, kwargs=kwargs
    )[0]
    if not kwargs:
        assert template['metadata']['kwargs'] == {}
    if not datasets:
        assert template['datasets'] == []
    if not args:
        assert template['metadata']['args'] == []
    assert template['type'] == type
    assert template['metadata']['method'] == method
    assert template['metadata']['version'] == version
