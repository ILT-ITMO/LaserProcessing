import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa

# Open the dataset
dataset = ds.dataset("D:\dataset\dataset.parquet", format="parquet")

# Count total rows efficiently
total_rows = dataset.scanner().count_rows()
limit_rows = int(total_rows * 0.8)

# Create scanner with small batch size
scanner = dataset.scanner(batch_size=10)
reader = scanner.to_reader()

# Initialize writer
writer = None
written_rows = 0

# Stream row batches from the reader
for batch in reader:
    num_rows = batch.num_rows

    if written_rows >= limit_rows:
        break

    if written_rows + num_rows <= limit_rows:
        if writer is None:
            writer = pq.ParquetWriter("C:\dataset\dataset_501.parquet", batch.schema)
        writer.write_table(pa.Table.from_batches([batch]))
        written_rows += num_rows
    else:
        # Slice batch to fit exactly
        remaining = limit_rows - written_rows
        partial_batch = batch.slice(0, remaining)
        if writer is None:
            writer = pq.ParquetWriter("C:\dataset\dataset_502.parquet", batch.schema)
        writer.write_table(pa.Table.from_batches([partial_batch]))
        written_rows += remaining
        break

# Finalize writer
if writer:
    writer.close()