#!/bin/bash


# --- Configuration ---

# --- Paths on THIS (Training) Server ---
PROJECT_DIR="/opt/feedback_module"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
ETL_SCRIPT="$PROJECT_DIR/fraudsms_data_etl.py" # ETL script
VALIDATE_SCRIPT="$PROJECT_DIR/train_and_validate.py" # validation & training script
LOG_FILE="/var/log/fraudsms_etl/retraining_pipeline.log"

# --- Details for the REMOTE Prediction Server ---
PREDICTION_SERVER_USER="ml_user" # The user on the remote prediction server
PREDICTION_SERVER_IP="10.15.94.141" 
PREDICTION_SERVER_DIR="/opt/realtime_fds" # Project dir on the remote prediction server


# --- Script Logic ---


# Simple logging function
log() {
    TIMESTAMP=$(date -u -d '+5 hours 30 minutes' +'%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] $1" | tee -a $LOG_FILE
}

log "------------------------------------"
log "Starting Retraining & Deployment Pipeline"
log "------------------------------------"

# --- 1. Run the ETL Script to find and add new data ---
log "[1/5] Running ETL script to find new reviewed data..."
# Change to the project directory to ensure scripts can find their files
cd $PROJECT_DIR || { log "❌ ERROR: Could not navigate to project directory $PROJECT_DIR. Aborting."; exit 1; }
# Activate venv, run script, and capture output
ETL_OUTPUT=$(source $PROJECT_DIR/venv/bin/activate && $VENV_PYTHON $ETL_SCRIPT)
echo "$ETL_OUTPUT" >> $LOG_FILE

# --- 2. Check if the ETL script appended new data ---
if echo "$ETL_OUTPUT" | grep -q "ETL_PROCESS_APPENDED_NEW_DATA"; then
    log "[2/5] New data found. Proceeding with model validation and training..."
    
    # --- 3. Run the Validation and Training Script ---
    log "[3/5] Starting validation and training script..."
    # The python script will print its own logs. We check its exit code.
    source $PROJECT_DIR/venv/bin/activate && $VENV_PYTHON $VALIDATE_SCRIPT >> $LOG_FILE 2>&1
    
    # Check the exit code of the validation script. 0 is success.
	if [ $? -eq 0 ]; then
		log "[4/5] ✅ Model validation successful. Deploying to prediction server..."
		
		# --- 5. Securely Copy new model files & restart remote service ---
		log "Copying new model files to ${PREDICTION_SERVER_IP}..."
		if scp final_model.joblib final_vectorizer.joblib \
			${PREDICTION_SERVER_USER}@${PREDICTION_SERVER_IP}:/tmp/; then
			
			log "Hotswapping model by restarting the remote service..."

			if ssh ${PREDICTION_SERVER_USER}@${PREDICTION_SERVER_IP} "\
				sudo mv /tmp/final_model.joblib /opt/realtime_fds/final_model.joblib && \
				sudo mv /tmp/final_vectorizer.joblib /opt/realtime_fds/final_vectorizer.joblib && \
				sudo systemctl restart fds_api"; then

				log "[5/5] ✅ Deployment complete."

			else
				log "❌ ERROR: Remote SSH commands failed. Deployment aborted."
				exit 1
			fi

		else
			log "❌ ERROR: Failed to copy model files to prediction server."
			exit 1
		fi
    else
        log "[4/5] ❌ Model validation failed. New model is not better. Deployment aborted."
    fi
else
    log "[2/5] No new data was found. Skipping training and deployment."
fi

log "Pipeline finished."
log "------------------------------------"
