"""Unit tests for architxt.bucket.zodb.ZODBTreeBucket class."""

import contextlib
import uuid
from collections.abc import AsyncIterable
from pathlib import Path

import anyio
import pytest
from architxt.bucket.zodb import ZODBTreeBucket
from architxt.tree import NodeLabel, NodeType, Tree
from hypothesis import given

from tests.test_strategies import tree_st


def create_test_trees(count: int) -> list[Tree]:
    """Create test trees for testing."""
    return [
        Tree(
            NodeLabel(NodeType.GROUP, f'test_group_{i}'),
            [Tree(NodeLabel(NodeType.ENT, f'entity_a_{i}'), []), Tree(NodeLabel(NodeType.ENT, f'entity_b_{i}'), [])],
        )
        for i in range(count)
    ]


async def async_tree_generator(trees: list[Tree]) -> AsyncIterable[Tree]:
    """Generate trees asynchronously."""
    for tree in trees:
        await anyio.lowlevel.checkpoint()
        yield tree


def test_open_existing_bucket(tmp_path: Path) -> None:
    """Test opening an existing bucket."""
    tree = Tree('test', [])

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        bucket1.add(tree)

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert len(bucket2) == 1
        assert tree in bucket2


def test_close_cleans_up_temp_dir() -> None:
    """Test that close() cleans up temporary directory."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket:
        bucket.add(tree)
        temp_path = bucket._storage_path
        assert temp_path.exists()

    # Temporary directory should be cleaned up
    assert not temp_path.exists()


def test_close_preserves_explicit_storage(tmp_path: Path) -> None:
    """Test that close() preserves explicit storage paths."""
    tree = Tree('test', [])

    with ZODBTreeBucket(storage_path=tmp_path) as bucket:
        bucket.add(tree)

    # Storage path should still exist
    assert tmp_path.exists()


@pytest.mark.parametrize("count", [0, 5, 100])
def test_update(count: int) -> None:
    """Test `update`."""
    trees = create_test_trees(count)

    with ZODBTreeBucket() as bucket:
        bucket.update(trees)

        assert len(bucket) == count
        assert all(tree in bucket for tree in trees), "All trees should be present after the update"


@pytest.mark.anyio
@pytest.mark.parametrize("count", [0, 5, 100])
@pytest.mark.parametrize("use_async_iterable", [False, True])
async def test_async_update(use_async_iterable: bool, count: int) -> None:
    """Test `async_update`."""
    trees = create_test_trees(count)
    iterable = async_tree_generator(trees) if use_async_iterable else trees

    with ZODBTreeBucket() as bucket:
        await bucket.async_update(iterable)

        assert len(bucket) == count
        assert all(tree in bucket for tree in trees), "All trees should be present after the update"


@pytest.mark.anyio
async def test_async_update_concurrent_safety() -> None:
    """Concurrent calls to `async_update` should all be applied (no lost updates)."""
    trees = create_test_trees(20)
    trees1 = trees[:10]
    trees2 = trees[10:]

    with ZODBTreeBucket() as bucket:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(bucket.async_update, trees1)
            task_group.start_soon(bucket.async_update, trees2)

        assert len(bucket) == 20
        assert all(tree in bucket for tree in trees), "All trees should be present after the update"


@given(tree=tree_st(has_parent=False))
def test_add_single_tree(tree: Tree) -> None:
    """Test adding a single tree."""
    with ZODBTreeBucket() as bucket:
        bucket.add(tree)

        assert len(bucket) == 1
        assert tree in bucket
        assert bucket[tree.oid] == tree


def test_discard_tree() -> None:
    """Test removing a tree."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket:
        bucket.add(tree)

        assert len(bucket) == 1
        assert tree in bucket

        bucket.discard(tree)
        assert len(bucket) == 0
        assert tree not in bucket


def test_clear_all_trees() -> None:
    """Test clearing all trees."""
    trees = create_test_trees(10)

    with ZODBTreeBucket() as bucket:
        bucket.update(trees)
        assert len(bucket) == 10

        bucket.clear()
        assert len(bucket) == 0


def test_get_tree_by_oid() -> None:
    """Test retrieving a tree by OID."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket:
        bucket.add(tree)

        retrieved = bucket[tree.oid]

        assert retrieved.oid == tree.oid
        assert retrieved.label == tree.label


def test_get_multiple_trees_by_oids() -> None:
    """Test retrieving multiple trees by OIDs."""
    trees = create_test_trees(5)

    with ZODBTreeBucket() as bucket:
        bucket.update(trees)

        oids = [t.oid for t in trees[:3]]

        retrieved = list(bucket[oids])
        assert len(retrieved) == 3

        retrieved_oids = [t.oid for t in retrieved]
        assert set(retrieved_oids) == set(oids)


def test_contains_tree() -> None:
    """Test checking if tree exists in bucket."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket:
        assert tree not in bucket

        bucket.add(tree)
        assert tree in bucket


def test_contains_oid() -> None:
    """Test checking if OID exists in bucket."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket:
        bucket.add(tree)

        assert tree.oid in bucket
        assert uuid.uuid4() not in bucket


def test_iteration() -> None:
    """Test iterating over trees in bucket."""
    trees = create_test_trees(5)

    with ZODBTreeBucket() as bucket:
        bucket.update(trees)

        result = list(bucket)

        assert len(result) == 5

        result_oids = {t.oid for t in result}
        original_oids = {t.oid for t in trees}
        assert result_oids == original_oids


def test_oids_generator() -> None:
    """Test OIDs generator."""
    trees = create_test_trees(5)

    with ZODBTreeBucket() as bucket:
        bucket.update(trees)

        oids = list(bucket.oids())
        assert len(oids) == 5

        original_oids = [t.oid for t in trees]
        assert set(oids) == set(original_oids)


def test_commit(tmp_path: Path) -> None:
    """Test explicit commit."""
    tree = Tree('test', [])
    oid = tree.oid

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        bucket1.add(tree)

        tree.label = 'modified'
        bucket1.commit()

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert bucket2[oid].label == 'modified', "Tree label should be persisted after commit."


def test_transaction(tmp_path: Path) -> None:
    """Test using transaction as context manager."""
    tree = Tree('test', [])
    oid = tree.oid

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        bucket1.add(tree)

        with bucket1.transaction():
            tree.label = 'modified'

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert bucket2[oid].label == 'modified', "Tree label should be persisted after transaction."


def test_no_commit(tmp_path: Path) -> None:
    """Test that changes without commit are not persisted."""
    tree = Tree('test', [])
    oid = tree.oid

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        bucket1.add(tree)
        tree.label = 'modified'

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert bucket2[oid].label == 'test', "Tree label should not be persisted without commit nor transaction."


def test_transaction_abort_on_exception(tmp_path: Path) -> None:
    """Test that transaction abort on exception."""
    tree = Tree('test', [])
    oid = tree.oid

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        bucket1.add(tree)

        with contextlib.suppress(ValueError), bucket1.transaction():
            tree.label = 'modified'
            raise ValueError

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert bucket2[oid].label == 'test', "Transaction should have abort on exception."


def test_duplicate_trees() -> None:
    """Test adding the same tree twice (should update, not duplicate)."""
    with ZODBTreeBucket() as bucket:
        tree = Tree(NodeLabel(NodeType.ENT, 'test'), [])

        bucket.add(tree)
        bucket.add(tree)  # Add again

        assert len(bucket) == 1, "Bucket should not contain duplicate trees."


def test_trees_with_same_oid() -> None:
    """Test handling trees with the same OID."""
    with ZODBTreeBucket() as bucket:
        oid = uuid.uuid4()
        tree1 = Tree('tree1', [], oid=oid)
        tree2 = Tree('tree2', [], oid=oid)

        bucket.add(tree1)
        bucket.add(tree2)  # Overwrites tree1

        assert len(bucket) == 1, "Bucket should contain only one tree with the same OID."
        assert bucket[oid].label == 'tree2', "Bucket should contain the last added tree with the same OID."


def test_empty_bucket_operations() -> None:
    """Test operations on an empty bucket."""
    with ZODBTreeBucket() as bucket:
        assert len(bucket) == 0
        assert list(bucket) == []
        assert list(bucket.oids()) == []

        # Test contains with non-existent tree
        fake_tree = Tree('fake', [])
        assert fake_tree not in bucket
