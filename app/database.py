from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

# The database URL for a local SQLite file
DATABASE_URL = os.getenv("DATABASE_URL")

# Create the async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create a session maker
async_session = async_sessionmaker(engine, expire_on_commit=False)

# Base class for our declarative models
Base = declarative_base() 