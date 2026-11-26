"""Independent entry point for the fixed EfficientNet B6 scalability experiment."""

from scalability_benchmark import cli


def main() -> None:
    cli.main(model_selection="model_variant")


if __name__ == "__main__":
    main()
