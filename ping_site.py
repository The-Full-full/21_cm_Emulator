import requests

url = "https://21cmvision.streamlit.app/"
try:
    response = requests.get(url)
    response.raise_for_status()  # מוודא שלא חזרה שגיאת רשת
    print(f"Successfully pinged the site! Status Code: {response.status_code}")
except Exception as e:
    print(f"Failed to ping the site: {e}")
