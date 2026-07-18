import sys
import time

COMPETITION = "playground-series-s5e7"
SUBMISSION_FILE = "data/processed/submission_stage2.csv"
SUBMISSION_MESSAGE = "Stage 2: mice + brm_multiple, outcome-free imputation"


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Submitting {SUBMISSION_FILE} to {COMPETITION}...")
    api.competition_submit(SUBMISSION_FILE, SUBMISSION_MESSAGE, COMPETITION)

    print("Waiting for the submission to score...")
    for _ in range(30):
        time.sleep(5)
        submissions = api.competition_submissions(COMPETITION)
        latest = submissions[0]
        if latest.status == "complete":
            print(f"\nPublic score: {latest.public_score}")
            print(f"Status: {latest.status}")
            print(f"Message: {latest.description}")
            return
        if latest.status == "error":
            print(f"\nSubmission errored: {latest.error_description}", file=sys.stderr)
            sys.exit(1)

    print("Timed out waiting for score; check the competition's Submissions page.")


if __name__ == "__main__":
    main()
