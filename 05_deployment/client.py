import requests

url = "http://localhost:9696/predict"

client_2 = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client_2).json()

print("Probability of converting = ")
print(response['converted_probability'])