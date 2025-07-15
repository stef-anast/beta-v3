import httpx
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date, timedelta

from . import crud, database


async def scrape_agones(url: str):
    """
    Asynchronously scrapes sports data from agones.gr for a given date.

    Args:
        url (str): The URL of the page to scrape.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped data,
                          or None if scraping fails.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()  # Raise an exception for bad status codes
    except httpx.RequestError as e:
        print(f"An HTTP error occurred while fetching {url}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"A bad status code was received for {url}: {e.response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the container div with id="results"
    results_div = soup.find('div', id='results')

    if not results_div:
        print("Could not find the results div with id='results'.")
        return None

    # Find the table within the div
    table = results_div.find('table')

    if not table:
        print("Could not find a data table inside div#results.")
        return None

    # Extract table headers
    headers = [th.get_text(strip=True) for th in table.find_all('th')]

    # Ensure the required odds columns are present in the table headers.
    required_odds = ['1', 'Χ', '2']
    if not all(h in headers for h in required_odds):
        print(f"Could not find one or more of the required odds columns {required_odds}.")
        return None

    # Extract table rows, ensuring they have odds data
    data = []
    for row in table.find_all('tr'):
        row_data = {}
        cells = row.find_all('td')
        if len(cells) == len(headers):
            # Process each cell based on its header
            for header, cell in zip(headers, cells):
                if header == 'ΟΜΑΔΕΣ':
                    team_divs = cell.find_all('div', class_='table--matches__team_row')
                    row_data['team_home'] = team_divs[0].get_text(strip=True) if len(team_divs) > 0 else ''
                    row_data['team_away'] = team_divs[1].get_text(strip=True) if len(team_divs) > 1 else ''
                else:
                    row_data[header] = cell.get_text(strip=True)

            # Check if all odds columns have a non-empty value
            has_odds = all(row_data.get(h, '').strip() for h in ['1', 'Χ', '2'])
            
            if has_odds and row_data.get('team_home') and row_data.get('team_away'):
                data.append(row_data)

    if not data:
        print("No rows with complete odds data found in the table.")
        return None

    # Create a pandas DataFrame from our list of dictionaries
    df = pd.DataFrame(data)
    return df


async def scrape_and_save_last_n_days(days: int):
    """
    Scrapes data for the last n days from yesterday and saves it to the database.
    """
    today = date.today()
    base_url = "https://agones.gr/ticker_minisite_show.php?navigation=yes&date="
    
    async with database.async_session() as db:
        for i in range(1, days + 1):
            current_date = today - timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")

            # Check if data already exists for this date before scraping
            if await crud.check_if_data_exists_for_date(db, current_date):
                print(f"Data for {date_str} already exists in DB. Skipping scrape.")
                continue

            url = f"{base_url}{date_str}"
            print(f"Scraping data for {date_str}...")
            
            try:
                daily_data = await scrape_agones(url)
                if daily_data is not None and not daily_data.empty:
                    daily_data['Date'] = date_str  # Add a date column
                    if 'ΣΚΟΡ' in daily_data.columns:
                        daily_data['Result'] = daily_data['ΣΚΟΡ'].apply(get_match_result)
                    
                    # Save the data for the day immediately
                    await crud.bulk_insert_matches(db, daily_data)
                else:
                    print(f"No data found for {date_str}.")
            except Exception as e:
                print(f"An error occurred while scraping for {date_str}: {e}")
                continue


def get_match_result(score_str: str):
    """
    Determines the match result from a score string (e.g., "1 - 0").

    Args:
        score_str (str): The score string.

    Returns:
        str: '1' for home win, '2' for away win, 'Χ' for a draw, or None.
    """
    try:
        # Clean up the score string from potential extra characters and split
        parts = score_str.strip().split(' - ')
        if len(parts) == 2:
            home_goals, away_goals = map(int, parts)
            if home_goals > away_goals:
                return '1'
            elif home_goals < away_goals:
                return '2'
            else:
                return 'Χ'
    except (ValueError, AttributeError):
        # Handles cases where conversion to int fails or input is not a string
        return None
    return None
