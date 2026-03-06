import pandas as pd
from sqlalchemy import create_engine, text
import datetime

# --- A. Custom Logging Function with Timezone ---
def log_message(message):
    """Prints a message with a timestamp in GMT+5:30."""
    # Get current UTC time
    utc_now = datetime.datetime.utcnow()
    # Create the +05:30 offset
    slt_offset = datetime.timedelta(hours=5, minutes=30)
    # Apply the offset to get Sri Lanka time
    slt_time = utc_now + slt_offset
    # Format the timestamp string
    timestamp = slt_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} +0530] {message}")


# --- CONFIGURATION ---
DATABASE_URL = "postgresql://feedback_user:Fds1584@localhost/feedback_db"
main_dataset_path = '/opt/feedback_module/final_dataset_v4.csv'

# --- SCRIPT LOGIC ---
def run_etl_pipeline():
    log_message("Starting data population pipeline for all reviewed SMS...")
    
    engine = create_engine(DATABASE_URL)
    connection = None  # Initialize connection to None
    
    try:
        connection = engine.connect()
        
        # 1. EXTRACT: Get all 'approved' AND 'rejected' messages
        log_message("Extracting 'approved' and 'rejected' messages from the database...")
        extract_query = "SELECT id, message_content, status FROM reported_sms WHERE status IN ('approved', 'rejected');"
        reviewed_df = pd.read_sql(extract_query, connection)
        
        if reviewed_df.empty:
            log_message("No new reviewed messages to process. Exiting.")
            return

        log_message(f"Found {len(reviewed_df)} new messages to add.")

        # 2. TRANSFORM: Format the data and assign correct labels
        log_message("Transforming data format...")
        processed_ids = tuple(reviewed_df['id'].tolist())
        
        new_data_df = pd.DataFrame()
        new_data_df['message'] = reviewed_df['message_content']
        new_data_df['label'] = reviewed_df['status'].apply(lambda x: 1 if x == 'approved' else 0)
        
        # 3. LOAD: Append the new data to the main CSV file
        log_message(f"Appending new data to '{main_dataset_path}'...")
        new_data_df.to_csv(main_dataset_path, mode='a', header=False, index=False)
        log_message("Append successful.")

        # 4. DELETE: Remove the processed records from the staging database
        log_message(f"Deleting {len(processed_ids)} processed records from 'reported_sms' table...")
        delete_query = text("DELETE FROM reported_sms WHERE id IN :ids")
        connection.execute(delete_query, {"ids": processed_ids})
        
        connection.commit()
        log_message("Deletion successful.")

    except Exception as e:
        log_message(f"An error occurred during the ETL process: {e}")
        if connection:
            connection.rollback() # Rollback changes if any error occurred
    
    finally:
        if connection:
            connection.close() # Always ensure the connection is closed
    
    log_message("Pipeline finished.")
    log_message("ETL_PROCESS_APPENDED_NEW_DATA")

if __name__ == "__main__":
    run_etl_pipeline()
