from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, APIRouter, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
from dotenv import load_dotenv
import logging
from starlette.requests import Request
from starlette.datastructures import State

from app import crud, ml_logic, scraper, models
from app.database import async_session, engine, Base

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Security ---
API_KEY = os.getenv("ADMIN_API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models ---
class TrainingRequest(BaseModel):
    days_to_use: int = Field(..., gt=0, description="Number of past days of data to use for training the model. Must be at least 1.")

# --- Background Task Logic ---
async def run_training_job(task_id: str, days_to_use: int, state: State):
    """
    The actual long-running function for scraping and training.
    This function now manages the global training lock via the app state.
    """
    try:
        state.task_statuses[task_id] = {"status": "in_progress", "message": f"Scraping the last {days_to_use} days."}
        # I/O-bound part (scraping, DB access) runs in the main async event loop
        async with async_session() as db:
            await scraper.scrape_and_save_last_n_days(days_to_use)
            df = await crud.get_all_matches_as_dataframe(db)
        
        if df.empty:
            raise ValueError("No historical data found after scraping. Cannot train model.")
        
        state.task_statuses[task_id]['message'] = "Data scraped. Starting model training in a separate process."
        
        # CPU-bound part (ML training) is run in the process pool
        loop = asyncio.get_running_loop()
        best_model = await loop.run_in_executor(
            state.process_pool, ml_logic.train_and_select_best_model, df
        )
        
        if not best_model:
            raise ValueError("Failed to train a model. Check logs for details.")

        async with async_session() as db:
            await crud.save_model(db, best_model)
        
        state.ml_models['best_model'] = best_model
        
        state.task_statuses[task_id] = {
            "status": "completed",
            "message": "Model training completed successfully and saved to database."
        }

    except Exception as e:
        logging.error(f"Training job {task_id} failed: {e}", exc_info=True)
        state.task_statuses[task_id] = {"status": "failed", "message": f"An error occurred: {str(e)}"}
    finally:
        # Release the lock
        state.training_job_running = False
        logging.info(f"Training job {task_id} finished. Lock has been released.")

# --- App Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, initialize the state and database
    app.state.process_pool = ProcessPoolExecutor()
    app.state.ml_models = {}
    app.state.task_statuses = {}
    app.state.training_job_running = False
    
    logging.info("--- API Starting Up: Initializing Database and State ---")
    
    async with engine.begin() as conn:
        # This ensures all tables are created based on the models.
        # It won't re-create tables that already exist.
        await conn.run_sync(Base.metadata.create_all)
    
    # Load the model from the database
    async with async_session() as db:
        app.state.ml_models['best_model'] = await crud.get_latest_model(db)

    if app.state.ml_models.get('best_model'):
        logging.info("Successfully loaded model from the database.")
    else:
        logging.warning("Could not load a model from the database. Please train a model.")
    
    yield
    
    # On shutdown, clean up
    logging.info("--- API Shutting Down: Cleaning up Process Pool ---")
    app.state.process_pool.shutdown(wait=True)

# --- FastAPI Application ---
app = FastAPI(
    title="Match Prediction API",
    description="An API to get football match predictions based on historical data and odds.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Dependency for getting a DB session ---
async def get_db():
    async with async_session() as session:
        yield session

# --- Main API Endpoints ---
@app.get("/")
def read_root():
    """A simple health-check endpoint to confirm the API is running."""
    return {"message": "Welcome to the Match Prediction API!"}

@app.post("/train-model", status_code=202)
async def train_model(
    request: TrainingRequest, 
    background_tasks: BackgroundTasks,
    app_request: Request,
    api_key: str = Depends(get_api_key)
):
    """
    Kicks off a background job to train a new model.
    Prevents starting a new job if one is already in progress.
    """
    if app_request.app.state.training_job_running:
        raise HTTPException(
            status_code=409,
            detail="A model training job is already in progress. Please wait for it to complete."
        )

    app_request.app.state.training_job_running = True
    task_id = str(uuid.uuid4())
    app_request.app.state.task_statuses[task_id] = {"status": "pending", "message": "Task received. Waiting to start."}
    
    background_tasks.add_task(run_training_job, task_id, request.days_to_use, app_request.app.state)
    
    return {
        "message": "Model training job started in the background.",
        "task_id": task_id,
        "status_endpoint": f"/train-model/status/{task_id}"
    }

@app.get("/train-model/status/{task_id}")
def get_training_status(task_id: str, request: Request):
    """Checks the status of a background training job."""
    status = request.app.state.task_statuses.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    return status

@app.get("/predictions/{match_date}")
async def get_predictions(match_date: str, request: Request, db: AsyncSession = Depends(get_db)):
    """
    Returns predictions for a given date.
    - Scrapes the data for the requested date to ensure it's up-to-date.
    - Saves the fresh data to the database, overwriting any old data for that date.
    - Uses the pre-trained model to make predictions on the new data.
    """
    try:
        parsed_date = datetime.strptime(match_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    if not request.app.state.ml_models.get('best_model'):
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first via the /train-model endpoint.")

    # Always scrape fresh data for the requested date
    logging.info(f"Scraping fresh data for {match_date}...")
    scraped_data = await scraper.scrape_agones(f"https://agones.gr/ticker_minisite_show.php?navigation=yes&date={match_date}")
    
    if scraped_data is None or scraped_data.empty:
        raise HTTPException(status_code=404, detail=f"Could not find or scrape any match data for {match_date}.")
    
    # Process and save the fresh data
    scraped_data['Date'] = match_date
    if 'ΣΚΟΡ' in scraped_data.columns:
        scraped_data['Result'] = scraped_data['ΣΚΟΡ'].apply(scraper.get_match_result)
    
    await crud.bulk_insert_matches(db, scraped_data)
    
    # Retrieve the freshly saved data to ensure consistency
    matches_df = await crud.get_matches_by_date_as_dataframe(db, parsed_date)

    if matches_df.empty:
        # This case should ideally not be reached if scraping was successful
        raise HTTPException(status_code=404, detail=f"No match data available for {match_date} after scraping.")

    odds_columns = ['odds_1', 'odds_x', 'odds_2']
    for col in odds_columns:
        matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce')
    
    predict_df = matches_df.dropna(subset=odds_columns).copy()
    if predict_df.empty:
        raise HTTPException(status_code=404, detail=f"No matches with valid odds found for {match_date}.")

    X_predict = predict_df[['odds_1', 'odds_x', 'odds_2']]
    
    model = request.app.state.ml_models['best_model']
    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)

    predict_df['Predicted_Result'] = predictions
    for i, class_label in enumerate(model.classes_):
        predict_df[f'Prob_{class_label}'] = probabilities[:, i]

    column_rename_map = {
        'match_date': 'matchDate',
        'match_code': 'matchCode',
        'team_home': 'teamHome',
        'team_away': 'teamAway',
        'odds_1': 'odds1',
        'odds_x': 'oddsX',
        'odds_2': 'odds2',
        'Predicted_Result': 'predictedResult',
        'Prob_1': 'prob1',
        'Prob_Χ': 'probX',
        'Prob_2': 'prob2'
    }
    predict_df.rename(columns=column_rename_map, inplace=True)

    final_columns = [
        'id', 'matchDate', 'competition', 'time', 'matchCode', 'teamHome', 
        'teamAway', 'odds1', 'oddsX', 'odds2', 'score', 'result', 
        'predictedResult', 'prob1', 'probX', 'prob2'
    ]
    existing_columns_to_return = [col for col in final_columns if col in predict_df.columns]

    return predict_df[existing_columns_to_return].to_dict(orient="records") 

# --- Admin Router ---
admin_router = APIRouter(
    prefix="/admin", 
    tags=["Admin"],
    dependencies=[Depends(get_api_key)]
)

@admin_router.post("/clear-data")
async def clear_data(db: AsyncSession = Depends(get_db)):
    """Deletes all records from the 'matches' table."""
    await crud.delete_all_matches(db)
    logging.info("Admin action: Match data cleared.")
    return {"message": "All match data has been cleared from the database."}

@admin_router.post("/clear-model")
async def clear_model(request: Request, db: AsyncSession = Depends(get_db)):
    """Deletes all saved models from the database and clears from app memory."""
    await crud.delete_all_models(db)
    
    if request.app.state.ml_models.get('best_model'):
        request.app.state.ml_models['best_model'] = None
    
    message = "All trained models have been cleared from the database and active memory."
    logging.info(f"Admin action: {message}")
    return {"message": message}

app.include_router(admin_router) 