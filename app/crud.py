import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime
from sqlalchemy import delete, update
import joblib
import io
from typing import Any

from . import models

# Mapping from scraped DataFrame columns to the database Model fields
COLUMN_MAPPING = {
    'Date': 'match_date',
    'ΔIO': 'competition',
    'ΩΡΑ': 'time',
    'ΚΩΔ': 'match_code',
    'team_home': 'team_home',
    'team_away': 'team_away',
    '1': 'odds_1',
    'Χ': 'odds_x',
    '2': 'odds_2',
    'ΣΚΟΡ': 'score',
    'Result': 'result'
}

async def bulk_insert_matches(db: AsyncSession, df: pd.DataFrame):
    """
    Takes a DataFrame of scraped matches, cleans it, and saves new entries to the database.
    It skips any matches that are already present.
    """
    # Rename columns to match the database model
    df_renamed = df.rename(columns=COLUMN_MAPPING)

    # --- Data Cleaning and Type Conversion ---
    df_renamed['match_date'] = pd.to_datetime(df_renamed['match_date']).dt.date
    for col in ['odds_1', 'odds_x', 'odds_2']:
        df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')

    new_matches_to_add = []
    for _, row in df_renamed.iterrows():
        # Create a dictionary of the row data that matches the model
        match_data = row.to_dict()
        # Filter out any columns/data that aren't in our model or are empty
        valid_columns = [c.name for c in models.Match.__table__.columns]
        filtered_data = {k: v for k, v in match_data.items() if k in valid_columns and pd.notna(v)}
        new_matches_to_add.append(models.Match(**filtered_data))

    if new_matches_to_add:
        db.add_all(new_matches_to_add)
        await db.commit()
        print(f"Successfully added {len(new_matches_to_add)} new matches to the database.")
    else:
        print("No new matches to add. Data may already be up to date.")


async def get_all_matches_as_dataframe(db: AsyncSession) -> pd.DataFrame:
    """
    Retrieves all matches from the database and returns them as a pandas DataFrame.
    """
    print("Fetching all historical match data from the database...")
    result = await db.execute(select(models.Match))
    all_matches = result.scalars().all()
    
    df = pd.DataFrame([match.__dict__ for match in all_matches])
    # The __dict__ approach includes SQLAlchemy internal state, so we drop it.
    df = df.drop(columns=['_sa_instance_state'], errors='ignore')
    
    print(f"Loaded {len(df)} matches into DataFrame for training.")
    return df

async def check_if_data_exists_for_date(db: AsyncSession, match_date: datetime.date) -> bool:
    """
    Checks if there is at least one match record for a specific date in the database.
    Returns True if data exists, False otherwise.
    """
    result = await db.execute(select(models.Match.id).filter(models.Match.match_date == match_date).limit(1))
    return result.first() is not None

async def get_matches_by_date_as_dataframe(db: AsyncSession, match_date: datetime.date) -> pd.DataFrame:
    """
    Retrieves all matches for a specific date from the database and returns them as a pandas DataFrame.
    """
    print(f"Fetching match data for {match_date} from the database...")
    result = await db.execute(select(models.Match).filter(models.Match.match_date == match_date))
    matches_for_date = result.scalars().all()

    if not matches_for_date:
        return pd.DataFrame()

    df = pd.DataFrame([match.__dict__ for match in matches_for_date])
    df = df.drop(columns=['_sa_instance_state'], errors='ignore')

    print(f"Loaded {len(df)} matches for {match_date} into DataFrame.")
    return df

async def delete_all_matches(db: AsyncSession):
    """
    Deletes all records from the matches table.
    """
    print("Deleting all match data from the database...")
    await db.execute(delete(models.Match))
    await db.commit()
    print("Database cleared.")

# --- CRUD Functions for Trained Models ---

async def save_model(db: AsyncSession, model: Any):
    """
    Serializes a trained model and saves it to the database.
    It also deactivates any previously active models.
    """
    # Deactivate all other models first
    await db.execute(
        update(models.TrainedModel)
        .where(models.TrainedModel.is_active == True)
        .values(is_active=False)
    )

    # Serialize the model object into bytes
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    model_bytes = buffer.read()

    # Create and save the new model instance
    new_model = models.TrainedModel(
        model_data=model_bytes,
        is_active=True
    )
    db.add(new_model)
    await db.commit()
    print(f"Successfully saved new model to the database.")

async def get_latest_model(db: AsyncSession) -> Any | None:
    """
    Retrieves the latest active model from the database and deserializes it.
    """
    print("Fetching the latest active model from the database...")
    result = await db.execute(
        select(models.TrainedModel)
        .filter(models.TrainedModel.is_active == True)
        .order_by(models.TrainedModel.created_at.desc())
    )
    latest_model_record = result.scalars().first()

    if not latest_model_record:
        print("No active model found in the database.")
        return None

    # Deserialize the model from bytes
    buffer = io.BytesIO(latest_model_record.model_data)
    model = joblib.load(buffer)
    print("Successfully loaded model from database.")
    return model

async def delete_all_models(db: AsyncSession):
    """
Deletes all records from the trained_models table.
    """
    print("Deleting all trained models from the database...")
    await db.execute(delete(models.TrainedModel))
    await db.commit()
    print("Trained models cleared from the database.") 