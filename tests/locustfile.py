"""
CamerTrust Lite - Test de charge avec Locust
E2 - Developpeur Backend | S8

Lancer localement (interface web) :
  locust -f tests/locustfile.py --host http://localhost:8000

Lancer contre l'API deployee sur Render :
  locust -f tests/locustfile.py --host https://TON-APP.onrender.com

Puis ouvrir http://localhost:8089, definir le nombre d'utilisateurs et
demarrer. Objectif du plan : latence P90 < 2s.
"""

import random

from locust import HttpUser, between, task

TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]


class CamerTrustUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def predict(self):
        payload = {
            "amount": round(random.uniform(500, 2_000_000), 2),
            "type": random.choice(TRANSACTION_TYPES),
            "old_balance_org": round(random.uniform(0, 5_000_000), 2),
            "new_balance_org": round(random.uniform(0, 5_000_000), 2),
            "old_balance_dest": round(random.uniform(0, 5_000_000), 2),
            "new_balance_dest": round(random.uniform(0, 5_000_000), 2),
        }
        self.client.post("/predict", json=payload)

    @task(1)
    def health(self):
        self.client.get("/health")
