import pandas as pd
import pyarrow.parquet as pq

# Define the file path
file_path = r"C:\Users\Onur Kılıç\Downloads\QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet"

try:
    # Initialize the Parquet file reader to inspect schema without loading the entire file into RAM
    parquet_file = pq.ParquetFile(file_path)
    
    print("-" * 30)
    print("File Schema (Column Names):", parquet_file.schema.names)
    print("-" * 30)

    # Load only the first 5 rows into a Pandas DataFrame to conserve memory
    df_preview = next(parquet_file.iter_batches(batch_size=5)).to_pandas()
    
    print("First 5 Rows Loaded Successfully:")
    print(df_preview.head())

    # Check the data structure of the 'X_jets' column if it exists
    if 'X_jets' in df_preview.columns:
        sample_jet = df_preview['X_jets'].iloc[0]
        print(f"Single jet data length/channels: {len(sample_jet)}")

except Exception as e:
    print(f"Error occurred during file reading: {e}")