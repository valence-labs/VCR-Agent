import pytest
import re

TEST_DATA_PATH = "/mnt/ps/home/CORP/emmanuel.noutahi/project/outgoing/hooke/hooke-explain/data/structure-explain-results-v2.csv"

@pytest.fixture(scope="module")
def test_data():
    data = pd.read_csv(TEST_DATA_PATH)
    return data


@pytest.fixture(scope="module")
def test_scl_data(test_data):
    scl_rows = [expl for row in test_data["explain"] for expl in row.split("\n") if "localises_to" in expl]

    # extract the scl info from explain
    scl_data = []
    for expl in scl_rows:
        entity = re.search(r'entity\s*["\']([^"\']+)["\']', expl).group(1)
        from_loc = re.search(r'from_loc\s*["\']([^"\']+)["\']', expl).group(1)
        to_loc = re.search(r'to_loc\s*["\']([^"\']+)["\']', expl).group(1)
        scl_data.append([entity, from_loc, to_loc])

    scl_data = pd.DataFrame(scl_data, columns = ["entity", "from_loc", "to_loc"])

    return scl_data