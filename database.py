from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging

DATABASE_CONNECTION_URL = os.getenv("DATABASE_CONNECTION_URL")

if not DATABASE_CONNECTION_URL:
    raise ValueError("DATABASE_CONNECTION_URL environment variable is not set or empty!")


logging.basicConfig(level=logging.INFO)

try:
    # Create the engine
    engine = create_engine(DATABASE_CONNECTION_URL)
    print("Database engine created successfully.")
except Exception as e:
    raise Exception(f"Failed to create database engine: {e}")

# Create session
try:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print("SessionLocal initialized successfully.")
except Exception as e:
    raise Exception(f"Failed to initialize SessionLocal: {e}")