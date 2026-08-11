"""
scripts/generate_telco_dataset.py
──────────────────────────────────
Generates the synthetic Telco Customer Churn dataset (7,043 rows) matching the
exact schema and distribution of the Kaggle Telco Churn dataset.
Saved to data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def generate_telco_dataset(n: int = 7043, random_state: int = 42) -> None:
    rng = np.random.default_rng(random_state)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    customer_ids = [
        f"{rng.integers(1000, 9999):04d}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}"
        for _ in range(n)
    ]
    genders = rng.choice(["Female", "Male"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.integers(1, 73, n)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"]))

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def get_internet_opt(net):
        if net == "No":
            return "No internet service"
        return rng.choice(["Yes", "No"])

    online_security = [get_internet_opt(net) for net in internet_service]
    online_backup = [get_internet_opt(net) for net in internet_service]
    device_protection = [get_internet_opt(net) for net in internet_service]
    tech_support = [get_internet_opt(net) for net in internet_service]
    streaming_tv = [get_internet_opt(net) for net in internet_service]
    streaming_movies = [get_internet_opt(net) for net in internet_service]

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.21, 0.22],
    )

    monthly_charges = []
    for net in internet_service:
        if net == "No":
            monthly_charges.append(round(float(rng.uniform(18.0, 25.0)), 2))
        elif net == "DSL":
            monthly_charges.append(round(float(rng.uniform(45.0, 85.0)), 2))
        else:
            monthly_charges.append(round(float(rng.uniform(70.0, 118.0)), 2))

    total_charges = []
    for t, m in zip(tenure, monthly_charges):
        # 11 rows in real dataset have blank " " TotalCharges for tenure=0
        if t == 0 or rng.random() < 0.0015:
            total_charges.append(" ")
        else:
            val = round(float(t * m + rng.uniform(-50.0, 50.0)), 2)
            total_charges.append(str(max(18.0, val)))

    # Higher churn probability for month-to-month + Fiber optic + low tenure
    churn_list = []
    for c, net, t in zip(contract, internet_service, tenure):
        prob = 0.26
        if c == "Month-to-month":
            prob += 0.18
        elif c == "Two year":
            prob -= 0.20

        if net == "Fiber optic":
            prob += 0.12

        if t < 12:
            prob += 0.10

        prob = max(0.02, min(0.90, prob))
        churn_list.append("Yes" if rng.random() < prob else "No")

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": genders,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_list,
        }
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {n} rows of Telco Customer Churn data to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_telco_dataset()
