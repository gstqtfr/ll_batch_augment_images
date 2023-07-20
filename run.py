import argparse
from pathlib import Path
from controller.workflow import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Aoply iage transformations to dataset")
    parser.add_argument("--yaml",
                        type=str,
                        required=True,
                        help="YAML file containing input and output directories fro images and labels")
    opts = parser.parse_args()
    yaml_file = opts.yaml
    if not Path(yaml_file).is_file():
        print(f"Cannot find YAML file  {yaml_file}")
        exit(0)
    run_pipeline(yaml_file)

if __name__  == "__main__":
    main()