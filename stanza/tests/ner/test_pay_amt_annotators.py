from stanza.utils import payingAnnotators

"""
Simple test for tracking AMT annotator work
"""


def test_amt_annotator_track():
    completed_amt_job_metadata = '.\\awscli-labeling'
    workers = {
        "7efc17ac-3397-4472-afe5-89184ad145d0": "Worker1",
        "afce8c28-969c-4e73-a20f-622ef122f585": "Worker2",
        "91f6236e-63c6-4a84-8fd6-1efbab6dedab": "Worker3",
        "6f202e93-e6b6-4e1d-8f07-0484b9a9093a": "Worker4",
        "2b674d33-f656-44b0-8f90-d70a1ab71ec2": "Worker5"
    }  # map AMT annotator subs to relevant identifier

    tracked_work = payingAnnotators.track_tasks(completed_amt_job_metadata, workers)
    assert tracked_work == {'Worker4': 948, 'Worker5': 865, 'Worker2': 134, 'Worker3': 735, 'Worker1': 18}


def test_amt_annotator_track_no_map():
    completed_amt_job_metadata = '.\\awscli-labeling'
    sub_to_count = payingAnnotators.track_tasks(completed_amt_job_metadata)
    assert sub_to_count == {'6f202e93-e6b6-4e1d-8f07-0484b9a9093a': 948, '2b674d33-f656-44b0-8f90-d70a1ab71ec2': 865,
            'afce8c28-969c-4e73-a20f-622ef122f585': 134, '91f6236e-63c6-4a84-8fd6-1efbab6dedab': 735,
            '7efc17ac-3397-4472-afe5-89184ad145d0': 18}


def main():
    test_amt_annotator_track()
    test_amt_annotator_track_no_map()


if __name__ == "__main__":
    main()
