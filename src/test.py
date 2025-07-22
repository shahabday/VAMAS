import yaml
from pathlib import Path
from batterydata.parsers.basytec_reader import BasyTecReader

# Load file path from config
CONFIG_FILE = Path(__file__).parent.parent / 'local_config.yaml'
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

file_path = config['basytec_sample_file']

# Create the reader and read data
reader = BasyTecReader(filepath=file_path, sample_id="test_sample_001")
result = reader.read()

# Print results
print("\n--- METADATA ---")
for k, v in result['metadata'].items():
    print(f"{k}: {v}")

print("\n--- DATA HEAD ---")
print(result['data'].head())

print("\n--- DATA COLUMNS ---")
print(result['data'].columns.tolist())
