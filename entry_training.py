import argparse
from src.entry_point.services.entry import entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd_path", type=str, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--test_every", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--target_epochs", type=int, required=True)
    args = parser.parse_args()
    entry(
        jd_path=args.jd_path,
        save_every=args.save_every,
        test_every=args.test_every,
        device=args.device,
        target_epochs=args.target_epochs,
    )
