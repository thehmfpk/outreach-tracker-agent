"""
generate_demo_leads.py
-----------------------
IMPORTANT: This produces SIMULATED / DEMO data only.
No real emails are sent. No real organizations are contacted.
This exists purely so the dashboard can *demonstrate* what an
automated lead-finder + outreach-tracker workflow would look like
in a live version of the product, for portfolio/demo purposes.

It generates 50 fictional small-business style leads (generic
Lahore-area business types/localities, not real named companies),
a templated outreach message per lead, a simulated send date, and
a simulated response status.

Run:
    python outreach/generate_demo_leads.py

Output:
    outreach/demo_leads.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)

business_types = [
    "Bakery", "Boutique", "Electronics Store", "Salon", "Restaurant",
    "Fitness Studio", "Furniture Shop", "Pharmacy", "Grocery Mart",
    "Auto Workshop", "Tuition Academy", "Print & Design Shop",
    "Mobile Accessories Store", "Cafe", "Tailoring Shop",
]

localities = [
    "Gulberg", "DHA Phase 5", "Johar Town", "Model Town", "Bahria Town",
    "Iqbal Town", "Township", "Faisal Town", "Wapda Town", "Cavalry Ground",
    "Garden Town", "Allama Iqbal Town", "Askari 10", "Valencia", "Sabzazar",
]

platforms = ["Email", "LinkedIn DM", "Instagram DM"]

responses = [
    "No response yet",
    "Interested - call scheduled",
    "Not interested - thanked us",
    "Asked for more info",
    "No response yet",
    "Interested - call scheduled",
]

response_weights = [0.42, 0.14, 0.16, 0.10, 0.10, 0.08]

MESSAGE_TEMPLATE = (
    "Hi {contact_name}, my name is {intern_name} and I'm an intern with "
    "SafeX Solutions. I'm reaching out because I'm building a small "
    "forecasting/analytics dashboard project and thought a business like "
    "{business_name} in {locality} could be a good fit for a short, "
    "no-obligation intro call about using simple demand forecasting for "
    "staffing/inventory planning. Totally fine to say no thanks - just let "
    "me know either way!"
)

intern_names = ["Ayesha", "Bilal", "Zainab", "Hamza", "Areeba"]

rows = []
start_date = datetime(2026, 7, 27)  # Week 4 start (simulated)

for i in range(1, 51):
    btype = np.random.choice(business_types)
    locality = np.random.choice(localities)
    business_name = f"{locality} {btype} #{i}"          # generic, non-identifying label
    contact_name = "Store Manager"                       # generic - no real person implied
    platform = np.random.choice(platforms)
    send_date = start_date + timedelta(days=int(np.random.randint(0, 5)))
    response = np.random.choice(responses, p=response_weights)
    intern = np.random.choice(intern_names)

    message = MESSAGE_TEMPLATE.format(
        contact_name=contact_name,
        intern_name=intern,
        business_name=business_name,
        locality=locality,
    )

    rows.append({
        "lead_id": i,
        "organization_demo_name": business_name,
        "business_type": btype,
        "locality_lahore": locality,
        "platform": platform,
        "contact_date": send_date.strftime("%Y-%m-%d"),
        "message_sent_demo": message,
        "response_status": response,
        "is_simulated": True,
    })

df = pd.DataFrame(rows)
df.to_csv("outreach/demo_leads.csv", index=False)
print(f"Generated {len(df)} SIMULATED demo leads -> outreach/demo_leads.csv")
