import requests

# ----------------------------
# Config
# ----------------------------
EC2_PUBLIC_IP = "YOUR_EC2_PUBLIC_IP"  # Replace with your EC2 public IP
USER_ID = 2                          # User you want recommendations for
TOP_K = 5                             # Number of recommendations

# ----------------------------
# Call endpoint
# ----------------------------
url = f"http://{EC2_PUBLIC_IP}:8000/recommend/{USER_ID}?top_k={TOP_K}"

try:
    response = requests.get(url)
    response.raise_for_status()  # Raise error if not 2xx
    recs = response.json()

    if not recs:
        print("No recommendations found for this user.")
    else:
        print(f"Top {TOP_K} recommendations for user {USER_ID}:")
        for r in recs:
            title = r.get("title", r["movieId"])
            score = r["hybrid_prediction"]
            print(f"- {title} (score: {score:.4f})")

except requests.exceptions.RequestException as e:
    print("Error calling the recommendation endpoint:", e)
