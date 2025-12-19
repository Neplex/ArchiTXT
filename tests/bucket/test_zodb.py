"""Unit tests for architxt.bucket.zodb.ZODBTreeBucket class."""

import contextlib
import multiprocessing
import uuid
from collections.abc import AsyncIterable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import anyio
import pytest
from architxt.bucket.zodb import ZODBTreeBucket
from architxt.tree import NodeLabel, NodeType, Tree, TreeOID
from architxt.utils import BATCH_SIZE
from hypothesis import given, settings

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

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1, bucket1.transaction():
        bucket1.add(tree)

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert len(bucket2) == 1, "Bucket should contain the tree when reopened."
        assert tree in bucket2


def test_close_cleans_up_temp_dir() -> None:
    """Test that close() cleans up temporary directory."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree)

        temp_path = bucket._storage_path
        assert temp_path.exists()

    # Temporary directory should be cleaned up
    assert not temp_path.exists()


def test_close_preserves_explicit_storage(tmp_path: Path) -> None:
    """Test that close() preserves explicit storage paths."""
    tree = Tree('test', [])

    with ZODBTreeBucket(storage_path=tmp_path) as bucket, bucket.transaction():
        bucket.add(tree)

    # Storage path should still exist
    assert tmp_path.exists()


@pytest.mark.parametrize("count", [0, 5, 100])
def test_update(count: int) -> None:
    """Test `update`."""
    trees = create_test_trees(count)

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
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
        with bucket.transaction():
            await bucket.async_update(iterable)

        assert len(bucket) == count
        assert all(tree in bucket for tree in trees), "All trees should be present after the update"


@pytest.mark.anyio
async def test_async_update_concurrent() -> None:
    """Concurrent calls to `async_update` should all be applied (no lost updates)."""
    trees = create_test_trees(20)
    trees1 = trees[:10]
    trees2 = trees[10:]

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(bucket.async_update, trees1)
                task_group.start_soon(bucket.async_update, trees2)

        assert len(bucket) == 20
        assert all(tree in bucket for tree in trees), "All trees should be present after the update"


@pytest.mark.anyio
async def test_async_update_concurrent_abort() -> None:
    """Concurrent calls to `async_update` should all be aborted on transaction abort."""
    trees = create_test_trees(20)
    trees1 = trees[:10]
    trees2 = trees[10:]

    with ZODBTreeBucket() as bucket:
        with contextlib.suppress(ValueError), bucket.transaction():
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(bucket.async_update, trees1)
                task_group.start_soon(bucket.async_update, trees2)
            raise ValueError  # <= Abort

        assert len(bucket) == 0, "Bucket should be empty after aborted transaction."


@settings(max_examples=10)
@given(tree=tree_st(has_parent=False))
def test_add_single_tree(tree: Tree) -> None:
    """Test adding a single tree."""
    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree)

        assert len(bucket) == 1
        assert tree in bucket
        assert bucket[tree.oid] == tree


def test_discard_tree() -> None:
    """Test removing a tree."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket, bucket.transaction():
        bucket.add(tree)

        assert len(bucket) == 1
        assert tree in bucket

        bucket.discard(tree)

        assert len(bucket) == 0
        assert tree not in bucket


def test_clear_all_trees() -> None:
    """Test clearing all trees."""
    trees = create_test_trees(10)

    with ZODBTreeBucket() as bucket, bucket.transaction():
        bucket.update(trees)
        assert len(bucket) == 10

        bucket.clear()
        assert len(bucket) == 0


def test_get_tree_by_oid() -> None:
    """Test retrieving a tree by OID."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket, bucket.transaction():
        bucket.add(tree)

        retrieved = bucket[tree.oid]

        assert retrieved.oid == tree.oid
        assert retrieved.label == tree.label


def test_get_multiple_trees_by_oids() -> None:
    """Test retrieving multiple trees by OIDs."""
    trees = create_test_trees(5)

    with ZODBTreeBucket() as bucket, bucket.transaction():
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

        with bucket.transaction():
            bucket.add(tree)

        assert tree in bucket


def test_contains_oid() -> None:
    """Test checking if OID exists in bucket."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket, bucket.transaction():
        bucket.add(tree)

        assert tree.oid in bucket
        assert uuid.uuid4() not in bucket


def test_iteration() -> None:
    """Test iterating over trees in bucket."""
    trees = create_test_trees(5)

    with ZODBTreeBucket() as bucket, bucket.transaction():
        bucket.update(trees)

        result = list(bucket)

        assert len(result) == 5

        result_oids = {t.oid for t in result}
        original_oids = {t.oid for t in trees}
        assert result_oids == original_oids


def test_oids_generator() -> None:
    """Test OIDs generator."""
    trees = create_test_trees(5)

    with ZODBTreeBucket() as bucket, bucket.transaction():
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
        with bucket1.transaction():
            bucket1.add(tree)

        with bucket1.transaction():
            tree.label = 'modified'

        assert bucket1[oid].label == 'modified', "Tree label should be modified."

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert bucket2[oid].label == 'modified', "Tree label should be persisted after commit."


def test_transaction_reentrant(tmp_path: Path) -> None:
    """Test reentrant transaction."""
    tree = Tree('test', [])
    oid = tree.oid

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        with bucket1.transaction():
            bucket1.add(tree)

        with bucket1.transaction(), bucket1.transaction(), bucket1.transaction():
            tree.label = 'modified'

        assert bucket1[oid].label == 'modified', "Tree label should be modified."

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert bucket2[oid].label == 'modified', "Tree label should be persisted after commit."


def test_abort(tmp_path: Path) -> None:
    """Test explicit abort."""
    tree = Tree('test', [])
    oid = tree.oid

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        with bucket1.transaction():
            bucket1.add(tree)

        with contextlib.suppress(ValueError), bucket1.transaction():
            tree.label = 'modified'
            raise ValueError  # <= Abort

        assert bucket1[oid].label == 'test', "Tree label should have been rollback."

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert bucket2[oid].label == 'test', "Tree label should not be persisted after rollback."


def test_update_abort_all(tmp_path: Path) -> None:
    """Test that transaction abort on exception."""

    def gen() -> Iterable[Tree]:
        for i in range(BATCH_SIZE * 4):
            yield Tree(f'tree{i}', [])
            if i == BATCH_SIZE * 2:
                msg = "abort"
                raise ValueError(msg)

    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        with pytest.raises(ValueError, match="abort"), bucket1.transaction():
            bucket1.update(gen())

        assert len(bucket1) == 0, "Bucket should be empty after aborted update."

    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        assert len(bucket2) == 0, "Transaction should have abort on exception."


def test_duplicate_trees() -> None:
    """Test adding the same tree twice (should update, not duplicate)."""
    with ZODBTreeBucket() as bucket, bucket.transaction():
        tree = Tree(NodeLabel(NodeType.ENT, 'test'), [])

        bucket.add(tree)
        bucket.add(tree)  # Add again

        assert len(bucket) == 1, "Bucket should not contain duplicate trees."


def test_trees_with_same_oid() -> None:
    """Test handling trees with the same OID."""
    with ZODBTreeBucket() as bucket, bucket.transaction():
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


def _worker(bucket: ZODBTreeBucket, label: str, oid: TreeOID) -> None:
    with bucket.transaction():
        tree = Tree(label, [])
        bucket.add(tree)

        tree = bucket[oid]
        tree.label = tree.label.replace('main', 'modified')


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_worker_sync(method: str) -> None:
    """Test that changes made in a worker process are visible after sync."""
    ctx = multiprocessing.get_context(method)
    tree_1 = Tree('main-1', [])
    tree_2 = Tree('main-2', [])

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree_1)
            bucket.add(tree_2)

        worker_1 = ctx.Process(target=_worker, args=(bucket, 'worker-1', tree_1.oid))
        worker_2 = ctx.Process(target=_worker, args=(bucket, 'worker-2', tree_2.oid))
        worker_1.start()
        worker_2.start()
        worker_1.join()
        worker_2.join()

        # ensure the workers exited cleanly
        assert worker_1.exitcode == 0
        assert worker_2.exitcode == 0

        # refresh connection of the DB and assert changes are visible
        bucket.sync()

        assert len(bucket) == 4, "Bucket should contain two trees"
        assert bucket[tree_1.oid].label == 'modified-1', "Tree label should be modified by worker"
        assert bucket[tree_2.oid].label == 'modified-2', "Tree label should be modified by worker"

        labels = [t.label for t in bucket]
        assert 'modified-1' in labels
        assert 'modified-2' in labels
        assert 'worker-1' in labels
        assert 'worker-2' in labels


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_worker_pool_sync(method: str) -> None:
    """Test that changes made in a worker process are visible after sync."""
    ctx = multiprocessing.get_context(method)
    tree_1 = Tree('main-1', [])
    tree_2 = Tree('main-2', [])

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree_1)
            bucket.add(tree_2)

        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
            worker_1 = pool.submit(_worker, bucket, 'worker-1', tree_1.oid)
            worker_2 = pool.submit(_worker, bucket, 'worker-2', tree_2.oid)

        # ensure the workers exited cleanly
        worker_1.result()
        worker_2.result()

        # refresh connection of the DB and assert changes are visible
        bucket.sync()

        assert len(bucket) == 4, "Bucket should contain two trees"
        assert bucket[tree_1.oid].label == 'modified-1', "Tree label should be modified by worker"
        assert bucket[tree_2.oid].label == 'modified-2', "Tree label should be modified by worker"

        labels = [t.label for t in bucket]
        assert 'modified-1' in labels
        assert 'modified-2' in labels
        assert 'worker-1' in labels
        assert 'worker-2' in labels


def test_threaded_worker_pool_sync() -> None:
    """Test that changes made in a worker thread are visible."""
    tree_1 = Tree('main-1', [])
    tree_2 = Tree('main-2', [])

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree_1)
            bucket.add(tree_2)

        with ThreadPoolExecutor(max_workers=2) as pool:
            worker_1 = pool.submit(_worker, bucket, 'worker-1', tree_1.oid)
            worker_2 = pool.submit(_worker, bucket, 'worker-2', tree_2.oid)

        # ensure the workers exited cleanly
        worker_1.result()
        worker_2.result()

        assert len(bucket) == 4, "Bucket should contain two trees"
        assert bucket[tree_1.oid].label == 'modified-1', "Tree label should be modified by worker"
        assert bucket[tree_2.oid].label == 'modified-2', "Tree label should be modified by worker"

        labels = [t.label for t in bucket]
        assert 'modified-1' in labels
        assert 'modified-2' in labels
        assert 'worker-1' in labels
        assert 'worker-2' in labels


def test_threaded_worker_pool_abort() -> None:
    """Test that changes made in a worker thread are visible."""
    tree_1 = Tree('main-1', [])
    tree_2 = Tree('main-2', [])

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree_1)
            bucket.add(tree_2)

        with contextlib.suppress(ValueError), bucket.transaction():
            with ThreadPoolExecutor(max_workers=2) as pool:
                worker_1 = pool.submit(_worker, bucket, 'worker-1', tree_1.oid)
                worker_2 = pool.submit(_worker, bucket, 'worker-2', tree_2.oid)
            raise ValueError  # <= Abort

        # ensure the workers exited cleanly
        worker_1.result()
        worker_2.result()

        assert len(bucket) == 2, "Bucket should contain two trees"
        assert bucket[tree_1.oid].label == 'main-1', "Tree label should not be modified after rollback"
        assert bucket[tree_2.oid].label == 'main-2', "Tree label should not be modified after rollback"

        labels = [t.label for t in bucket]
        assert 'main-1' in labels
        assert 'main-2' in labels
        assert 'worker-1' not in labels
        assert 'worker-2' not in labels


@given(tree=tree_st())
@settings(max_examples=20)
def test_persistent_ref(tree: Tree) -> None:
    """Property-based test for get_persistent_ref with various tree structures."""
    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree.root)

        ref = bucket.get_persistent_ref(tree)
        resolved = bucket.resolve_ref(ref)

        assert resolved is tree
        assert resolved.root is tree.root


def test_get_persistent_ref_not_stored() -> None:
    """Test getting persistent reference for a tree not in the bucket raises KeyError."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket, pytest.raises(KeyError, match="not stored in the bucket"):
        bucket.get_persistent_ref(tree)


def test_resolve_ref_invalid_reference() -> None:
    """Test resolving an invalid or fabricated reference."""
    with ZODBTreeBucket() as bucket, pytest.raises(KeyError, match="not stored in the bucket"):
        bucket.resolve_ref('invalid_reference')


def test_resolve_ref_after_modification() -> None:
    """Test resolving a reference after the tree has been modified."""
    tree = Tree('original', [])

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree)

        ref = bucket.get_persistent_ref(tree)

        # Modify the tree
        with bucket.transaction():
            tree.label = 'modified'

        # Resolve the reference - should get the modified version
        resolved = bucket.resolve_ref(ref)
        assert resolved.label == 'modified'


def test_resolve_ref_persists_across_connections(tmp_path: Path) -> None:
    """Test that persistent references work across different bucket connections."""
    tree = Tree('test', ['data'])
    ref = None

    # First connection: add tree and get reference
    with ZODBTreeBucket(storage_path=tmp_path) as bucket1:
        with bucket1.transaction():
            bucket1.add(tree)

        ref = bucket1.get_persistent_ref(tree)

    # Second connection: resolve the reference
    with ZODBTreeBucket(storage_path=tmp_path) as bucket2:
        resolved = bucket2.resolve_ref(ref)
        assert resolved == tree


def test_resolve_ref_unique_to_bucket() -> None:
    """Test that persistent references work across different bucket connections."""
    tree = Tree('test', ['data'])
    ref = None

    # First connection: add tree and get reference
    with ZODBTreeBucket() as bucket1:
        with bucket1.transaction():
            bucket1.add(tree)

        ref = bucket1.get_persistent_ref(tree)

    # Second connection: resolve the reference
    with ZODBTreeBucket() as bucket2, pytest.raises(KeyError, match="not stored in the bucket"):
        bucket2.resolve_ref(ref)


def test_resolve_ref_with_tree_removed() -> None:
    """Test resolving a reference after the tree has been removed from bucket."""
    tree = Tree('test', [])

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree)

        ref = bucket.get_persistent_ref(tree)

        # Remove the tree
        with bucket.transaction():
            bucket.discard(tree)

        # The tree object is removed from bucket
        with pytest.raises(KeyError, match="not stored in the bucket"):
            bucket.resolve_ref(ref)


def test_persistent_ref_unique_with_same_oid() -> None:
    """Test persistent references when multiple trees share the same OID."""
    oid = uuid.uuid4()
    tree = Tree(
        'ROOT',
        [
            Tree('1', [], oid=oid),
            Tree('2', [], oid=oid),
        ],
    )

    with ZODBTreeBucket() as bucket:
        with bucket.transaction():
            bucket.add(tree)

        ref1 = bucket.get_persistent_ref(tree[0])
        ref2 = bucket.get_persistent_ref(tree[1])
        assert ref1 != ref2

        resolved1 = bucket.resolve_ref(ref1)
        resolved2 = bucket.resolve_ref(ref2)
        assert resolved1.label == '1'
        assert resolved2.label == '2'
