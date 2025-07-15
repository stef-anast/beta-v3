import pandas as pd
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

from .scraper import scrape_agones


def train_and_select_best_model(df: pd.DataFrame):
    """
    Trains multiple models, evaluates them, and selects the best one.

    Args:
        df (pd.DataFrame): DataFrame containing the historical match data.

    Returns:
        The best-performing trained model object, or None if training fails.
    """
    odds_columns = ['odds_1', 'odds_x', 'odds_2']
    target_column = 'result' # Column names are lowercase from the DB model

    # The dataframe from the DB already has numeric types where possible,
    # but we coerce just in case of any data issues.
    for col in odds_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Column names in the dataframe from the DB are lowercase
    df_cleaned = df.dropna(subset=odds_columns + [target_column])

    if df_cleaned.shape[0] < 20:  # Increased minimum for robust evaluation
        print("Not enough clean data to train and evaluate models.")
        return None

    print(f"\nTraining models with {df_cleaned.shape[0]} matches...")
    print("Outcome distribution in training data:")
    # Use lowercase 'result' column from the DB
    print(df_cleaned[target_column].value_counts(normalize=True))

    X = df_cleaned[odds_columns]
    y = df_cleaned[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    best_accuracy = 0.0
    best_model = None
    best_model_name = ""

    print("\n--- Model Performance ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {accuracy:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print(f"\nBest performing model is '{best_model_name}' with an accuracy of {best_accuracy:.2f}.")

    print("\nClassification Report for Best Model:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))

    return best_model


def get_predictions_for_date(match_date: str, model):
    """
    Scrapes a given date's matches and predicts their outcomes using the trained model.

    Args:
        match_date (str): The date to scrape in 'YYYY-MM-DD' format.
        model: The trained classifier model.
    """
    print(f"\n--- Getting predictions for {match_date} ---")
    url = f"https://agones.gr/ticker_minisite_show.php?navigation=yes&date={match_date}"

    todays_data = scrape_agones(url)

    if todays_data is None or todays_data.empty:
        print("No matches found for the date or failed to scrape.")
        return None

    # --- Prepare data for prediction ---
    # The raw scraped data uses these column names
    odds_columns = ['1', 'Χ', '2']
    for col in odds_columns:
        todays_data[col] = pd.to_numeric(todays_data[col], errors='coerce')

    # Filter for rows that have valid odds
    predict_df = todays_data.dropna(subset=odds_columns).copy()

    if predict_df.empty:
        print("No matches with valid odds found for today.")
        return None

    X_today = predict_df[odds_columns]

    # --- Make Predictions ---
    predictions = model.predict(X_today)
    probabilities = model.predict_proba(X_today)

    # Add predictions to the DataFrame for display
    predict_df['Predicted_Result'] = predictions
    # Add probabilities for more insight
    for i, class_label in enumerate(model.classes_):
        predict_df[f'Prob_{class_label}'] = probabilities[:, i]

    return predict_df 