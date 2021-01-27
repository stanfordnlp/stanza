"""
Test some pieces of the depparse dataloader
"""
import pytest
from stanza.models.depparse.data import data_to_batches

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def make_fake_data(*lengths):
    data = []
    for i, length in enumerate(lengths):
        word = chr(ord('A') + i)
        chunk = [[word] * length]
        data.append(chunk)
    return data

def check_batches(batched_data, expected_sizes, expected_order):
    for chunk, size in zip(batched_data, expected_sizes):
        assert sum(len(x[0]) for x in chunk) == size
    word_order = []
    for chunk in batched_data:
        for sentence in chunk:
            word_order.append(sentence[0][0])
    assert word_order == expected_order

def test_data_to_batches_eval_mode():
    """
    Tests the chunking of batches in eval_mode

    A few options are tested, such as whether or not to sort and the maximum sentence size
    """
    data = make_fake_data(1, 2, 3)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 1], ['C', 'B', 'A'])

    data = make_fake_data(1, 2, 6)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [6, 3], ['C', 'B', 'A'])

    data = make_fake_data(3, 2, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 1], ['A', 'B', 'C'])

    data = make_fake_data(3, 5, 2)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 5], ['B', 'A', 'C'])

    data = make_fake_data(3, 5, 2)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [3, 5, 2], ['A', 'B', 'C'])

    data = make_fake_data(4, 1, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [4, 2], ['A', 'B', 'C'])

    data = make_fake_data(1, 4, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [1, 4, 1], ['A', 'B', 'C'])

if __name__ == '__main__':
    test_data_to_batches()

