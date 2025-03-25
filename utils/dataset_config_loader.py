import yaml


def load_config(config_path):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Parsed configuration data.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def usage():
    """Prints usage instructions for loading dataset configuration."""
    print("Usage: Load dataset configuration from a YAML file.")
    print("Example:")
    print("    config = load_config('/path/to/config.yml')")
    print("    print(config)")


if __name__ == "__main__":
    usage()
