import random
import pandas as pd

random.seed(42)

# ---- templates for each category ----
billing_templates = [
    "I was charged twice for my subscription this month, please refund me",
    "My invoice amount does not match what I was quoted",
    "I want to cancel my subscription and get a refund",
    "There is a wrong charge on my credit card from your company",
    "Payment failed but the amount was still deducted from my account",
    "Can you explain why my bill increased this month",
    "I need a copy of my last 3 invoices for tax purposes",
    "Auto renewal charged me even after I cancelled the plan",
    "asking for a discount code that isnt working at checkout",
    "billing cycle date is wrong on my account",
]

technical_templates = [
    "The app keeps crashing every time I try to open it",
    "I am getting a 500 error when I try to login to the dashboard",
    "Video upload is stuck at 99 percent since yesterday",
    "The website is completely down and showing a blank page",
    "Getting error code 403 forbidden when accessing my project",
    "Data is not syncing between mobile app and web app",
    "Login page keeps loading forever and never opens",
    "The export to pdf feature is broken and generates empty files",
    "Server keeps timing out when i upload large files",
    "app crashed and i lost all my unsaved work",
]

account_templates = [
    "I forgot my password and the reset link is not working",
    "I cannot access my account after changing my email",
    "My account got suspended without any warning email",
    "I need to change the email address linked to my account",
    "two factor authentication code never arrives on my phone",
    "please delete my account and all my personal data",
    "I want to merge two accounts into one",
    "my profile picture and username reset on their own",
    "unable to verify my account with the otp sent",
    "someone else might have access to my account, please help",
]

general_templates = [
    "Just wanted to know what plans you offer for students",
    "Do you have an API for third party integration",
    "What are your customer support working hours",
    "Is there a mobile app available for android users",
    "Can I get a demo of the enterprise plan",
    "How do I upgrade from the free plan to premium",
    "Where can I find your terms of service document",
    "Do you offer bulk pricing for teams larger than 50 people",
    "just checking if there is a referral program",
    "loved the new update, when is the next feature release",
]

urgent_words = ["urgent", "asap", "immediately", "right now", "critical",
                "cannot work", "losing money", "production is down", "emergency"]
medium_words = ["soon as possible", "please help", "waiting since 2 days",
                "still not fixed", "annoying"]

categories = {
    "Billing": billing_templates,
    "Technical Issue": technical_templates,
    "Account": account_templates,
    "General Query": general_templates,
}


def build_ticket():
    category = random.choice(list(categories.keys()))
    base_text = random.choice(categories[category])
    roll = random.random()

    if category == "General Query":
        priority = "Low" if roll < 0.85 else "Medium"
    elif category in ("Technical Issue", "Billing"):
        if roll < 0.4:
            priority = "High"
            base_text = base_text + " " + random.choice(urgent_words)
        elif roll < 0.75:
            priority = "Medium"
            base_text = base_text + " " + random.choice(medium_words)
        else:
            priority = "Low"
    else:  
        if roll < 0.25:
            priority = "High"
            base_text = base_text + " " + random.choice(urgent_words)
        elif roll < 0.65:
            priority = "Medium"
        else:
            priority = "Low"

    filler = random.choice(["", " thanks", " please advise", " regards",
                             " this has been going on for a while", ""])
    text = base_text + filler
    return text, category, priority


rows = [build_ticket() for _ in range(900)]
df = pd.DataFrame(rows, columns=["ticket_text", "category", "priority"])
df.insert(0, "ticket_id", range(1, len(df) + 1))

df.to_csv("support_tickets.csv", index=False)
print(df.shape)
print(df["category"].value_counts())
print(df["priority"].value_counts())