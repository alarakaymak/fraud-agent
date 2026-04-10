"""
Creates the two DynamoDB tables and seeds user_history with mock data.
Run once before deploying:
    python setup_dynamodb.py
"""

import boto3
import os
import time
import json
from decimal import Decimal

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
HISTORY_TABLE  = "fraud-user-history"
DECISIONS_TABLE = "fraud-agent-decisions"

dynamodb = boto3.resource("dynamodb", region_name=REGION)
client   = boto3.client("dynamodb",   region_name=REGION)

NOW = int(time.time())

MOCK_HISTORY = {
    # user_001: low spender, always in Nashville
    "user_001": [
        {"amount": 23.50,  "merchant": "Whole Foods",      "time": "08:30 AM", "city": "Nashville, TN", "status": "approved", "timestamp": NOW - 86400 * 2},
        {"amount": 156.20, "merchant": "Shell Gas Station", "time": "05:15 PM", "city": "Nashville, TN", "status": "approved", "timestamp": NOW - 86400 * 3},
        {"amount": 45.00,  "merchant": "Chipotle",          "time": "07:45 PM", "city": "Nashville, TN", "status": "approved", "timestamp": NOW - 86400 * 4},
        {"amount": 12.99,  "merchant": "Netflix",           "time": "09:00 AM", "city": "Nashville, TN", "status": "approved", "timestamp": NOW - 86400 * 5},
        {"amount": 89.99,  "merchant": "Gap",               "time": "02:30 PM", "city": "Nashville, TN", "status": "approved", "timestamp": NOW - 86400 * 6},
    ],
    # user_002: high spender, always in San Francisco
    "user_002": [
        {"amount": 4200.00, "merchant": "Best Buy",         "time": "11:00 AM", "city": "San Francisco, CA", "status": "approved", "timestamp": NOW - 3600 * 3},
        {"amount": 3800.00, "merchant": "Apple Store",      "time": "03:20 PM", "city": "San Francisco, CA", "status": "approved", "timestamp": NOW - 3600 * 6},
        {"amount": 2900.00, "merchant": "Samsung",          "time": "01:45 PM", "city": "San Francisco, CA", "status": "approved", "timestamp": NOW - 86400},
        {"amount": 500.00,  "merchant": "Amazon",           "time": "10:00 AM", "city": "San Francisco, CA", "status": "approved", "timestamp": NOW - 86400 * 2},
        {"amount": 1200.00, "merchant": "Dell",             "time": "04:00 PM", "city": "San Francisco, CA", "status": "approved", "timestamp": NOW - 86400 * 3},
    ],
    # user_003: small spender, burst of 4 transactions in last 10 minutes — card testing pattern
    "user_003": [
        {"amount": 15.00,  "merchant": "Starbucks",         "time": "07:00 AM", "city": "Chicago, IL", "status": "approved", "timestamp": NOW - 300},
        {"amount": 8.50,   "merchant": "McDonald's",        "time": "07:02 AM", "city": "Chicago, IL", "status": "approved", "timestamp": NOW - 480},
        {"amount": 22.00,  "merchant": "CVS Pharmacy",      "time": "07:04 AM", "city": "Chicago, IL", "status": "approved", "timestamp": NOW - 540},
        {"amount": 65.00,  "merchant": "Uber",              "time": "07:06 AM", "city": "Chicago, IL", "status": "approved", "timestamp": NOW - 590},
        {"amount": 30.00,  "merchant": "Trader Joe's",      "time": "07:08 AM", "city": "Chicago, IL", "status": "approved", "timestamp": NOW - 610},
    ],
}


def create_table_if_not_exists(name, key_schema, attribute_definitions, billing="PAY_PER_REQUEST"):
    existing = client.list_tables()["TableNames"]
    if name in existing:
        print(f"  Table '{name}' already exists — skipping.")
        return dynamodb.Table(name)

    table = dynamodb.create_table(
        TableName=name,
        KeySchema=key_schema,
        AttributeDefinitions=attribute_definitions,
        BillingMode=billing,
    )
    print(f"  Creating '{name}'...", end="", flush=True)
    table.wait_until_exists()
    print(" done.")
    return table


def main():
    print("Setting up DynamoDB tables...\n")

    # 1. User history table — PK: user_id, SK: timestamp (for ordering)
    history_table = create_table_if_not_exists(
        HISTORY_TABLE,
        key_schema=[
            {"AttributeName": "user_id",   "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"},
        ],
        attribute_definitions=[
            {"AttributeName": "user_id",   "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "N"},
        ],
    )

    # 2. Decisions table — PK: transaction_id (unique per decision)
    create_table_if_not_exists(
        DECISIONS_TABLE,
        key_schema=[
            {"AttributeName": "transaction_id", "KeyType": "HASH"},
        ],
        attribute_definitions=[
            {"AttributeName": "transaction_id", "AttributeType": "S"},
        ],
    )

    # Seed user history
    print(f"\nSeeding '{HISTORY_TABLE}'...")
    for user_id, txns in MOCK_HISTORY.items():
        for txn in txns:
            history_table.put_item(Item={
                "user_id":   user_id,
                "timestamp": txn["timestamp"],
                "amount":    Decimal(str(txn["amount"])),
                "merchant":  txn["merchant"],
                "time":      txn["time"],
                "city":      txn["city"],
                "status":    txn["status"],
            })
        print(f"  Seeded {len(txns)} transactions for {user_id}")

    print("\nDone. Set these env vars before running the agent or Lambda:")
    print(f"  DYNAMODB_HISTORY_TABLE={HISTORY_TABLE}")
    print(f"  DYNAMODB_DECISIONS_TABLE={DECISIONS_TABLE}")


if __name__ == "__main__":
    main()
