import json
import os

# Sample of JSON object returned from AWS:
# sample_json = {"answers": [{"acceptanceTime": "2022-08-05T06:44:45.745Z", "answerContent": {
#     "crowd-entity-annotation": {"entities": [{"endOffset": 80, "label": "Miscellaneous", "startOffset": 73}]}},
#                             "submissionTime": "2022-08-05T06:44:56.111Z", "timeSpentInSeconds": 10.366,
#                             "workerId": "private.us-east-1.47a0115a1f1786e9",
#                             "workerMetadata": {"identityData": {"identityProviderType": "Cognito",
#                                                                 "issuer": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_5BJRp9lN7",
#                                                                 "sub": "7efc17ac-3397-4472-afe5-89184ad145d0"}}}]}

# Ties workers to AWS Worker IDs, sample below
workers = {
        "7efc17ac-3397-4472-afe5-89184ad145d0": "Worker1",
        "afce8c28-969c-4e73-a20f-622ef122f585": "Worker2",
        "91f6236e-63c6-4a84-8fd6-1efbab6dedab": "Worker3",
        "6f202e93-e6b6-4e1d-8f07-0484b9a9093a": "Worker4",
        "2b674d33-f656-44b0-8f90-d70a1ab71ec2": "Worker5"
        }


def get_worker_subs(json_string):
    """
    Gets the AWS worker IDs from the annotation file in output folder.

    Returns a list of the AWS worker subs
    """
    subs = []
    # json.loads() works on JSON strings, json.load() is for JSON files
    job_data = json.loads(json_string)
    for i in range(len(job_data["answers"])):
        subs.append(job_data["answers"][i]["workerMetadata"]["identityData"]["sub"])
    return subs


def track_tasks(input_path, worker_map):
    """
    Takes a path to a folder containing the worker annotation metadata from AWS Sagemaker labeling job and a
    dictionary mapping AWS worker subs to their names or identification tags and returns a dictionary mapping
    the names/identification tags to the number of labeling tasks completed.

    :param input_path: string of the path to the directory containing the worker annotation sub-directories
    :param worker_map: dictionary mapping AWS worker subs to the worker identifications
    :return: dictionary mapping worker identifications to the number of tasks completed
    """
    tracker = {}
    res = {}
    for direc in os.listdir(input_path):
        subdir = os.listdir(input_path + "\\" + direc)
        json_file_path = input_path + "\\" + direc + "\\" + subdir[0]
        json_file = open(json_file_path)
        json_string = json_file.read()
        subs = get_worker_subs(json_string)
        for sub in subs:
            if sub not in tracker:
                tracker[sub] = 0
            tracker[sub] += 1
    for sub in tracker:
        worker = worker_map[sub]
        res[worker] = tracker[sub]
    return res


def main():
    # sample from completed labeling job
    print(track_tasks('C:\\Users\\Alex\\Desktop\\awscli-labeling', worker_map={
        "7efc17ac-3397-4472-afe5-89184ad145d0": "Worker1",
        "afce8c28-969c-4e73-a20f-622ef122f585": "Worker2",
        "91f6236e-63c6-4a84-8fd6-1efbab6dedab": "Worker3",
        "6f202e93-e6b6-4e1d-8f07-0484b9a9093a": "Worker4",
        "2b674d33-f656-44b0-8f90-d70a1ab71ec2": "Worker5"
        }
    ))
    return


if __name__ == "__main__":
    main()
