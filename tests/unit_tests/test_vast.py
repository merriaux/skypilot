"""Unit tests for the Vast provisioner."""
import pytest

from sky.provision.vast import utils


def _query_terms(instance_type: str,
                 region: str,
                 disk_size: int = 100,
                 secure_only: bool = False) -> list[str]:
    return utils._build_offer_query(instance_type, region, disk_size,
                                    secure_only).split(' ')


def test_build_offer_query_gpu_name_uses_underscore_form():
    """The GPU filter must use Vast's underscore name (e.g. `RTX_4090`).

    Vast's `parse_query` turns `_` back into a space; emitting a literal space
    (`gpu_name=RTX 4090`) instead makes the upstream tokenizer stop parsing and
    silently drop every remaining filter, so the search returns an arbitrary
    GPU.
    """
    terms = _query_terms('1x-RTX_4090-32-65536', 'Oregon, US, NA')
    assert 'gpu_name=RTX_4090' in terms
    assert 'gpu_name=RTX 4090' not in ' '.join(terms)

    terms_ada = _query_terms('1x-RTX_6000-Ada-48-131072', 'Oregon, US, NA')
    assert 'gpu_name=RTX_6000-Ada' in terms_ada


@pytest.mark.parametrize('region', ['Oregon, US, NA', ', US, NA', 'US', ''])
def test_build_offer_query_does_not_pin_geolocation(region: str):
    """No clause may carry the catalog row's region.

    The catalog keeps one snapshot row per (gpu, count) tuple, so its region is
    an artifact of when the snapshot was taken. Pinning on it rejects every live
    offer in any other country: with the pin, one RTX 5090 offer had the disk we
    asked for; without it, twelve did.
    """
    terms = _query_terms('1x-RTX_4090-32-65536', region)
    assert not [term for term in terms if term.startswith('geolocation')]
    assert not [term for term in terms if term.startswith('georegion')]


def test_build_offer_query_cpu_ram_is_integer_gb_lower_bound():
    """`cpu_ram` is multiplied by 1000 (GB) by Vast, so emit integer GB.

    The instance type encodes MiB (65536), which is 64 GB. A float like `64.0`
    would be truncated at the `.` by the upstream tokenizer and drop the rest
    of the query, so the value must be an int.
    """
    terms = _query_terms('1x-RTX_4090-32-65536', 'Oregon, US, NA')
    assert 'cpu_ram>=64' in terms


def test_build_offer_query_has_no_quotes_or_decimals():
    """Quotes and decimals make Vast's tokenizer stop and drop later filters."""
    query = utils._build_offer_query('1x-RTX_4090-32-65536', 'Oregon, US, NA',
                                     100, True)
    assert '"' not in query
    assert '.' not in query


def test_build_offer_query_num_gpus():
    terms = _query_terms('2x-RTX_4090-32-65536', 'Oregon, US, NA')
    assert 'num_gpus=2' in terms


def test_build_offer_query_disk_size():
    terms = _query_terms('1x-RTX_4090-32-65536',
                         'Oregon, US, NA',
                         disk_size=250)
    assert 'disk_space>=250' in terms


def test_build_offer_query_secure_only_adds_datacenter_filters():
    insecure = utils._build_offer_query('1x-RTX_4090-32-65536',
                                        'Oregon, US, NA', 100, False)
    assert 'datacenter=true' not in insecure
    assert 'hosting_type>=1' not in insecure

    secure = utils._build_offer_query('1x-RTX_4090-32-65536', 'Oregon, US, NA',
                                      100, True)
    assert 'datacenter=true' in secure
    assert 'hosting_type>=1' in secure


def test_build_offer_query_survives_real_vast_parser():
    """End-to-end check against Vast's own query parser, if installed.

    This is the regression that the unit-level string assertions cannot catch:
    Vast silently drops malformed filters instead of erroring, so we feed the
    query through the same `preprocess_search_query` + `parse_query` pipeline
    Vast uses server-side and assert each filter actually survives.
    """
    try:
        import vastai  # noqa: F401  # pylint: disable=import-outside-toplevel
    except (ImportError, SyntaxError):
        pytest.skip('vastai not available or requires Python 3.10+')
    # pylint: disable=import-outside-toplevel
    from vastai.api.query import offers_alias
    from vastai.api.query import offers_fields
    from vastai.api.query import offers_mult
    from vastai.api.query import parse_query
    from vastai.utils import preprocess_search_query

    query = utils._build_offer_query('1x-RTX_4090-32-65536', 'Maryland, US, NA',
                                     100, False)
    _, _, preprocessed = preprocess_search_query(query)
    parsed = parse_query(preprocessed, {}, offers_fields, offers_alias,
                         offers_mult)

    assert parsed['gpu_name'] == {'eq': 'RTX 4090'}
    assert parsed['num_gpus'] == {'eq': '1'}
    assert 'geolocation' not in parsed
    assert parsed['disk_space'] == {'gte': '100'}
    # cpu_ram is 64 GB, multiplied by 1000 by the parser.
    assert parsed['cpu_ram'] == {'gte': 64000.0}
    # The parser reads days and stores seconds: 3 -> 259200.
    assert parsed['duration'] == {
        'gte': float(utils._MIN_RENTAL_DAYS) * 86400
    }


def test_build_offer_query_requires_a_minimum_rental_window():
    """Hosts that commit to a short window stop the instance mid-job.

    Vast stops, without warning, when the advertised window ends; one offer
    committing to 1.8 days ended a 42-hour run.
    """
    terms = _query_terms('1x-RTX_4090-32-65536', 'Oregon, US, NA')
    assert f'duration>={utils._MIN_RENTAL_DAYS}' in terms
    # Days, not seconds: seconds parse fine and match no offer at all.
    assert utils._MIN_RENTAL_DAYS < 365


def test_register_ssh_key_attaches_to_the_instance(monkeypatch):
    """The key must be registered with Vast, not only appended by onstart.

    Vast owns the instance's authorized set and drops what it did not register,
    which strands a running cluster with `Permission denied (publickey)`.
    """
    calls = []

    class _Client:

        def attach_ssh(self, instance_id, ssh_key):
            calls.append((instance_id, ssh_key))
            return {'success': True}

    monkeypatch.setattr(utils.vast, 'vast', lambda: _Client())
    utils._register_ssh_key(42, '  ssh-rsa AAAA  ')
    assert calls == [(42, 'ssh-rsa AAAA')]


def test_register_ssh_key_warns_instead_of_raising(monkeypatch, caplog):
    """A launch that works now must not abort, but the failure must be visible.

    Registering on the account fails silently for team API keys (upstream
    #9631); repeating that mistake here would hide the same outage.
    """

    class _Client:

        def attach_ssh(self, instance_id, ssh_key):
            raise RuntimeError('boom')

    monkeypatch.setattr(utils.vast, 'vast', lambda: _Client())
    with caplog.at_level('WARNING'):
        utils._register_ssh_key(42, 'ssh-rsa AAAA')
    assert 'boom' in caplog.text
