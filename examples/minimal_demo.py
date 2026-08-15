"""Small end-to-end demonstration with synthetic visual probabilities."""

from rgvspd import decide_for_shot


def main() -> None:
    result = decide_for_shot(
        class_ids=["0", "1", "2", "3", "4", "5"],
        dino_probabilities=[0.45, 0.30, 0.10, 0.07, 0.05, 0.03],
        clip_probabilities=[0.20, 0.50, 0.12, 0.08, 0.06, 0.04],
        shot=2,
        semantic_votes={
            "attrseek": "0",
            "attrbank": "0",
            "finedefics": "1",
        },
    )
    print(f"routed={result.routed}")
    print(f"candidates={list(result.candidates)}")
    print(f"visual_prediction={result.visual_prediction}")
    print(f"final_prediction={result.prediction}")
    print(f"reason={result.reason}")


if __name__ == "__main__":
    main()
