import pandas as pd
import boto3
from io import StringIO
from sqlalchemy import create_engine, MetaData, Table, Column, String
from db.session import db_url
from utils.log import logger
import os

# S3 Configuration
S3_BUCKET = "parkstreet-ai"
S3_PREFIX = "csv-files/"  # Optional prefix/folder in S3 bucket

# Schema to use for the tables
SCHEMA = "public"


def create_db_engine():
    """Create and return a SQLAlchemy engine using existing db_url."""
    engine = create_engine(db_url)
    logger.info("Connected to PostgreSQL database using existing db_url")
    return engine


def get_s3_client():
    """Create and return a boto3 S3 client."""
    # Uses AWS credentials from environment or ~/.aws/credentials
    return boto3.client("s3")


def get_csv_files_from_s3(s3_client, bucket):
    """Get list of CSV files in the specified S3 bucket/prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket)

    if "Contents" not in response:
        logger.warning(f"No files found in s3://{bucket}")
        return []

    # Filter to only CSV files and extract just the filename (not full path)
    csv_files = []
    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith(".csv"):
            # Get just the filename without the prefix path
            filename = os.path.basename(key)
            if filename:  # Ensure it's not empty (would happen if key ends with /)
                csv_files.append({"key": key, "filename": filename})

    logger.info(f"Found {len(csv_files)} CSV files in s3://{bucket}")
    return csv_files


def read_csv_from_s3(s3_client, bucket, key, **kwargs):
    """Read a CSV file from S3 into a pandas DataFrame."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    csv_content = response["Body"].read().decode("utf-8")

    # Create a file-like object
    csv_buffer = StringIO(csv_content)

    # Read CSV into DataFrame
    return pd.read_csv(csv_buffer, **kwargs)


def infer_table_structure(s3_client, bucket, key, engine, schema=SCHEMA):
    """
    Infer table structure from S3 CSV file.
    Returns a SQLAlchemy Table object.
    """
    # Read just the header to get column names
    df_header = read_csv_from_s3(s3_client, bucket, key, nrows=0)

    # Create metadata object with schema
    metadata = MetaData(schema=schema)

    # Extract table name from file name (remove .csv extension)
    filename = os.path.basename(key)
    table_name = os.path.splitext(filename)[0].lower()

    # Create column definitions - use all String types to avoid type conversion issues
    columns = []
    for column in df_header.columns:
        # Clean column name (replace spaces with underscores, make lowercase)
        col_name = column.lower().replace(" ", "_")
        # Use String type for all columns for initial loading
        columns.append(Column(col_name, String))

    # Create table
    table = Table(table_name, metadata, *columns)

    # Create table in database if it doesn't exist
    metadata.create_all(engine)

    logger.info(f"Created table structure for {schema}.{table_name} with {len(columns)} columns")

    return table, table_name


def import_csv_to_db(s3_client, bucket, key, engine, schema=SCHEMA, problematic_columns=None):
    """
    Import data from S3 CSV file to PostgreSQL database table.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key (file path)
        engine: SQLAlchemy engine
        schema: Database schema
        problematic_columns: List of column names to drop
    """
    _, table_name = infer_table_structure(s3_client, bucket, key, engine, schema)

    try:
        # First read to get column names
        df_sample = read_csv_from_s3(s3_client, bucket, key, nrows=0)
        column_names = list(df_sample.columns)

        # Create a dictionary to force all columns to be read as strings
        dtype_dict = {col: str for col in column_names}

        # Read the full CSV with all columns as strings
        df = read_csv_from_s3(
            s3_client,
            bucket,
            key,
            dtype=dtype_dict,
            low_memory=False,
            na_values=["", "#N/A", "NULL"],
            keep_default_na=True,
        )

        # Clean column names (replace spaces with underscores, make lowercase)
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Drop problematic columns if specified
        if problematic_columns:
            # Convert column names to lowercase with underscores for matching
            formatted_problem_cols = [col.lower().replace(" ", "_") for col in problematic_columns]
            # Drop columns that exist in the dataframe
            cols_to_drop = [col for col in formatted_problem_cols if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped problematic columns: {', '.join(cols_to_drop)}")

        # Import to database - force all columns to be treated as strings
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            dtype={col: String for col in df.columns},
            chunksize=10000,  # Add chunking for large files
        )

        logger.info(
            f"Successfully imported {len(df)} rows from s3://{bucket}/{key} to table {schema}.{table_name}"
        )
        return True
    except Exception as e:
        logger.error(f"Error importing {key}: {str(e)}")

        # Try again with dropping the problematic column if identified in the error message
        if "column" in str(e) and "is of type" in str(e):
            # Extract column name from error message
            error_msg = str(e)
            start_idx = error_msg.find('column "') + 8
            end_idx = error_msg.find('" is of type')
            if start_idx > 7 and end_idx > start_idx:
                problem_column = error_msg[start_idx:end_idx]
                logger.info(f"Attempting to retry import without problematic column: {problem_column}")

                # Create or extend the list of problematic columns
                if problematic_columns:
                    problematic_columns.append(problem_column)
                else:
                    problematic_columns = [problem_column]

                # Retry the import without the problematic column
                return import_csv_to_db(s3_client, bucket, key, engine, schema, problematic_columns)

        # If no specific column identified or other error, give up
        return False


def main():
    """Main function to process all S3 CSV files and import to PostgreSQL."""
    # Create database engine using existing db_url
    engine = create_db_engine()

    # Create S3 client
    s3_client = get_s3_client()

    # Get list of CSV files from S3
    csv_files = get_csv_files_from_s3(s3_client, S3_BUCKET)

    # Define known problematic columns to drop (you can add to this list)
    known_problematic_columns = {
        "orders.csv": ["transaction_type"],
        # Add more files and their problematic columns as needed
    }

    # Process each CSV file independently
    success_count = 0
    for file_info in csv_files:
        key = file_info["key"]
        filename = file_info["filename"]
        logger.info(f"Processing {filename} (S3 key: {key})...")

        # Get predefined problematic columns for this file if any
        problem_cols = known_problematic_columns.get(filename, None)

        # Try to import with problematic columns dropped
        if import_csv_to_db(s3_client, S3_BUCKET, key, engine, problematic_columns=problem_cols):
            success_count += 1

    # Summary log
    if success_count == len(csv_files):
        logger.info(f"All {len(csv_files)} CSV files processed successfully!")
    else:
        logger.info(f"Processed {success_count} out of {len(csv_files)} CSV files successfully.")


if __name__ == "__main__":
    main()
