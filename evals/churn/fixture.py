"""Small deterministic synthetic subscription dataset for churn evaluations.

This fixture is generated from code instead of copied from a third-party source,
so it has no external dataset license or customer data attached to it.
"""
from datetime import date

import pandas as pd


SNAPSHOT_DATE = date(2026, 6, 30)


def _shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    absolute_month = year * 12 + month - 1 + offset
    return absolute_month // 12, absolute_month % 12 + 1


def build_fixture() -> pd.DataFrame:
    """Build 60 synthetic customers plus one exact duplicate row."""
    plans = ("monthly", "annual", "two_year")
    regions = ("East", "West")
    payment_methods = ("card", "bank_transfer", "invoice")
    rows = []

    for customer_number in range(1, 61):
        plan = plans[(customer_number - 1) % len(plans)]
        months_before_snapshot = (customer_number * 5) % 24
        signup_year, signup_month = _shift_month(
            SNAPSHOT_DATE.year, SNAPSHOT_DATE.month, -months_before_snapshot
        )
        current_tenure = max(
            1,
            (SNAPSHOT_DATE.year - signup_year) * 12
            + SNAPSHOT_DATE.month - signup_month,
        )
        churned = (
            (plan == "monthly" and customer_number % 2 == 0)
            or (plan == "annual" and customer_number % 5 == 0)
            or (plan == "two_year" and customer_number % 10 == 0)
        )
        tenure = (
            max(1, min(current_tenure, 1 + customer_number % max(current_tenure, 1)))
            if churned else current_tenure
        )
        churn_year, churn_month = _shift_month(signup_year, signup_month, tenure - 1)
        churn_date = date(churn_year, churn_month, 1).isoformat() if churned else None
        fee = round(
            {"monthly": 59.0, "annual": 49.0, "two_year": 39.0}[plan]
            + (customer_number % 7) * 1.5,
            2,
        )
        support_tickets = (customer_number * 7) % 6
        if churned:
            support_tickets = min(8, support_tickets + 1)

        rows.append({
            "customer_id": f"S{customer_number:04d}",
            "signup_date": date(signup_year, signup_month, 1).isoformat(),
            "signup_cohort": f"{signup_year}-Q{(signup_month - 1) // 3 + 1}",
            "churn_date": churn_date,
            "churned": churned,
            "plan": plan,
            "monthly_fee": None if customer_number in {17, 44} else fee,
            "tenure_months": tenure,
            "support_tickets": None if customer_number == 23 else support_tickets,
            "region": regions[customer_number % 2],
            "payment_method": payment_methods[(customer_number * 2) % 3],
            "auto_renew": plan != "monthly" and customer_number % 4 != 0,
        })

    rows.append(rows[1].copy())  # One deliberate exact duplicate for C14/C20.
    frame = pd.DataFrame(rows)
    frame["churned"] = frame["churned"].astype("boolean")
    frame["auto_renew"] = frame["auto_renew"].astype("boolean")
    frame["tenure_months"] = frame["tenure_months"].astype("Int64")
    frame["support_tickets"] = frame["support_tickets"].astype("Int64")
    frame["monthly_fee"] = frame["monthly_fee"].astype("Float64")
    return frame


DATA_DICTIONARY = """\
customer_id: synthetic customer identifier; never use as a predictive feature.
signup_date: date the subscription began.
signup_cohort: calendar quarter of signup.
churn_date: cancellation date for churned customers; blank for active customers.
churned: current binary churn outcome (true means churned).
plan: subscription term (monthly, annual, two_year).
monthly_fee: recurring monthly price; includes two deliberately missing values.
tenure_months: observed customer tenure in months.
support_tickets: support ticket count; includes one deliberately missing value.
region: East or West.
payment_method: card, bank_transfer, or invoice.
auto_renew: whether the subscription is enrolled in automatic renewal.
Snapshot date: 2026-06-30. This is one synthetic status snapshot plus signup
and churn dates; it does not include plan-change events or monthly billing history.
"""
