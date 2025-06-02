import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Path to the folder containing the binary files
input_folder = "D:/results/"
output_parquet = "D:/dataset/dataset.parquet"

# Define the schema for the Parquet file
schema = pa.schema([
    ("Repetition rate, kHz", pa.float32()),
    ("Energy, uJ", pa.float32()),
    ("Beam radius, mkm", pa.float32()),
    ("Pulse duration, ns", pa.float32()),
    ("Temperature Field, C", pa.binary())
])

# Create the Parquet writer
with pq.ParquetWriter(output_parquet, schema) as writer:
    for filename in os.listdir(input_folder):
        print(filename)
        if not filename.startswith("T_"):
            continue

        base = os.path.splitext(filename)[0]
        parts = base.split("_")

        if len(parts) != 5:
            print(f"Skipping malformed filename: {filename}")
            continue

        try:
            _, repetition_rate, energy, beam_radius, pulse_duration = parts

            file_path = os.path.join(input_folder, filename)
            with open(file_path, "rb") as f:
                binary_data = f.read()

            # Create a record batch with a single row
            batch = pa.record_batch([
                [float(repetition_rate)],
                [float(energy)],
                [float(beam_radius)],
                [float(pulse_duration)],
                [binary_data]
            ], schema=schema)

            # Write the batch (single row) to file
            writer.write_table(pa.Table.from_batches([batch]))

        except Exception as e:
            print(f"Error processing {filename}: {e}")