from collections.abc import Generator

import pytest
from sqlalchemy import Connection, create_engine


@pytest.fixture(name='connection')
def get_connection() -> Generator[Connection, None, None]:
    with create_engine('sqlite:///:memory:').connect() as connection:
        yield connection
