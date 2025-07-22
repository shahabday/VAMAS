# src/batterydata/parsers/basytec_reader.py

import pandas as pd
from batterydata.mappings.basytec_mapping import (
    BASYTEC_COLUMN_MAPPING,
    BASYTEC_METADATA_MAPPING,
)
class BasyTecReader:
    def __init__(self, filepath: str, sample_id: str = None):
        self.filepath = filepath
        self.sample_id = sample_id

    def read(self):
        metadata = self._extract_metadata()
        data = self._extract_data()
        data = self._map_columns_to_ontology(data)
        if self.sample_id:
            metadata['sample_id'] = self.sample_id
        return {"data": data, "metadata": metadata}

    def _extract_metadata(self):
        metadata = {}
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if ':' in line: 
                    key, value = line[1:].split(':', 1)
                    key_clean = key.strip().lower().replace(' ', '_')
                    # Use mapping for standard key names
                    std_key = BASYTEC_METADATA_MAPPING.get(key_clean, key_clean)
                    metadata[std_key] = value.strip()
        return metadata

    def _extract_data(self):
        # Find first line not starting with '#' (column headers), then read data as DataFrame
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_idx = None
        for idx, line in enumerate(lines):
            if not line.startswith('#'):
                header_idx = idx
                break
        df = pd.read_csv(
            self.filepath,
            sep=",",
            skiprows=header_idx,
            engine="python"
        )
        return df

    def _map_columns_to_ontology(self, df):
        df = df.rename(columns=BASYTEC_COLUMN_MAPPING)
        return df
