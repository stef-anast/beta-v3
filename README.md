# Match Prediction API

> **Live Demo:** [https://beta-v3-zs2y.onrender.com/](https://beta-v3-zs2y.onrender.com/)

> **Note:** This project is the successor to the original [beta-v2 (Flask-based) project](https://github.com/dai16240/beta-v2). This new version is a complete rewrite using modern technologies like FastAPI, asynchronous scraping, and a more advanced model selection pipeline.

This project is a complete, high-performance web API that scrapes football match data, trains multiple machine learning models to find the best one, and provides predictions for future matches based on betting odds.

It is built with Python using FastAPI and includes a robust architecture for handling long-running background tasks, database integration with SQLAlchemy, and a non-blocking scraping process.

## Key Features

- **Asynchronous Web Scraping**: Uses `httpx` to efficiently scrape match data from `agones.gr` without blocking the API server.
- **Intelligent Caching**: Scraped data is stored in a PostgreSQL database to avoid re-scraping the same data, making subsequent requests much faster.
- **Automated Model Selection**: Trains multiple classification models (Logistic Regression, Random Forest, SVC, etc.) and automatically selects the best-performing one based on accuracy.
- **Non-Blocking Background Jobs**: The resource-intensive model training process is run as a background job, allowing the API to remain responsive. A status endpoint is provided to check the progress of the training job.
- **Administrative Endpoints**: Includes endpoints for clearing the database and the saved model, making development and testing easier.
- **Automatic API Documentation**: Leverages FastAPI's built-in Swagger UI for interactive API documentation.
- **Persistent Model Storage**: The trained machine learning model is stored directly in the PostgreSQL database, ensuring it persists across deployments without relying on a filesystem.

### Security

- **API Key Authentication**: The `/train-model` and `/admin` routes are protected. Requests to these endpoints must include a valid `X-API-Key` header.

## Project Structure

## How It Works

This application is designed with a clear separation of concerns, making it modular and easy to maintain. Here is a breakdown of how the different parts of the code work together:

1.  **`main.py`: The API Layer**

    - This is the entry point of the application, powered by **FastAPI**.
    - It defines all the API endpoints, such as `/train-model`, `/predictions/{match_date}`, and the `/admin` routes.
    - It handles incoming requests, validates them using Pydantic models, and calls the appropriate logic from other modules.
    - It manages the application's lifecycle (`lifespan`), which includes initializing the database tables on startup and loading the latest trained model from the database.
    - For long-running tasks like model training, it uses **`BackgroundTasks`** to avoid blocking the server, immediately returning a task ID to the client.

2.  **`app/scraper.py`: Data Collection**

    - This module is responsible for fetching raw data from `agones.gr`.
    - The `scrape_agones` function uses `httpx` for asynchronous HTTP requests and `BeautifulSoup` to parse the HTML and extract match data into a structured format (a pandas DataFrame).
    - `scrape_and_save_last_n_days` orchestrates the scraping process, checking the database first to avoid re-scraping data that has already been collected.

3.  **`app/database.py`, `app/models.py`, `app/crud.py`: The Database Layer**

    - **`database.py`** sets up the connection to the PostgreSQL database using **SQLAlchemy's** asynchronous engine.
    - **`models.py`** defines the database schema using SQLAlchemy models. There are two main tables: `matches` for storing game data and `trained_models` for storing the serialized machine learning models.
    - **`crud.py`** (Create, Read, Update, Delete) contains all the functions that interact with the database. It handles tasks like inserting new matches, retrieving historical data for training, saving a newly trained model, and fetching the latest active model. This keeps all SQL logic separate from the main application logic.

4.  **`app/ml_logic.py`: The Machine Learning Core**
    - This is where the "magic" happens.
    - `train_and_select_best_model` takes the historical match data as a DataFrame.
    - It then defines a set of different classification models (like Logistic Regression, Random Forest, etc.) and trains each one on the data.
    - After training, it evaluates the models based on their accuracy and automatically selects the best-performing one.
    - The best model is then returned, ready to be saved to the database and used for future predictions.

This modular architecture ensures that the API remains fast and responsive while the heavy lifting of scraping and model training happens efficiently in the background.

```
├── app/
│   ├── __init__.py
│   ├── crud.py         # Database Create, Read, Update, Delete functions
│   ├── database.py     # SQLAlchemy database engine and session setup
│   ├── ml_logic.py     # Core machine learning training and prediction logic
│   ├── models.py       # SQLAlchemy table models
│   └── scraper.py      # Web scraping and data parsing logic
├── create_db.py        # One-time script to initialize the database
├── main.py             # Main FastAPI application, defines all API endpoints
├── best_model.pkl      # The saved file for the best-performing trained model
├── .env                # Environment variables for database configuration
└── requirements.txt    # Project dependencies
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your environment:**
    Copy the example `.env.dist` file to a new `.env` file. This file will hold your local environment variables and should not be committed to source control.

    ```bash
    # On Windows (Command Prompt)
    copy .env.dist .env

    # On macOS/Linux
    cp .env.dist .env
    ```

    Now, open the `.env` file and replace the placeholder values with your actual PostgreSQL database URL and a strong, randomly generated Admin API Key.

5.  **Run the API server:**
    The first time you run the application, it will automatically create the necessary database tables.

    ```bash
    uvicorn main:app --reload
    ```

    The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

The interactive API documentation (Swagger UI) is the best way to explore and test the endpoints. Once the application is running, you can access it at:

- **Locally:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Live Demo:** [https://beta-v3-zs2y.onrender.com/docs](https://beta-v3-zs2y.onrender.com/docs)

### Training

- **`POST /train-model`**: Kicks off a background job to train a new model.

  - **Authentication**: Requires `X-API-Key` header.
  - **Request Body**: `{"days_to_use": 90}` (specify the number of past days to use for training data, must be at least 1).
  - **Response**: Immediately returns a `task_id` to check the job status.

- **`GET /train-model/status/{task_id}`**: Checks the status of a training job.
  - **Response**: Shows the status (`pending`, `in_progress`, `completed`, `failed`) and any relevant messages.

### Predictions

- **`GET /predictions/{match_date}`**: Gets predictions for a specific date.
  - **URL Parameter**: `match_date` in `YYYY-MM-DD` format.
  - **Response**: A JSON array of match objects with predicted outcomes and probabilities.

### Administration

All admin routes require a valid `X-API-Key` in the request header.

- **`POST /admin/clear-data`**: Deletes all match records from the database.
- **`POST /admin/clear-model`**: Deletes all trained models from the database and clears the active model from memory.

## Running Tests

This project uses `pytest` for unit testing. To run the test suite, ensure you have installed the development dependencies from `requirements.txt`.

Then, from the root of the project directory, simply run:

```bash
pytest
```

The tests will run, and you will see a summary of the results. The tests are designed to run in isolation and will mock database connections and other external services, so you do not need a live database to run them.
