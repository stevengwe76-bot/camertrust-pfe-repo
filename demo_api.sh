#!/usr/bin/env bash
# CamerTrust Lite - Script de demo de l'API via curl
# Usage : ./demo_api.sh https://camertrust-api.onrender.com
#         (par defaut : http://localhost:8000)

set -e

BASE_URL="${1:-http://localhost:8000}"

echo "=== 1. Health check ==="
curl -s "$BASE_URL/health" | python3 -m json.tool

echo -e "\n=== 2. Infos sur le modele ==="
curl -s "$BASE_URL/info" | python3 -m json.tool

echo -e "\n=== 3. Prediction sur une transaction suspecte ==="
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "amount": 1850000,
        "type": "TRANSFER",
        "old_balance_org": 1850000,
        "new_balance_org": 0,
        "old_balance_dest": 0,
        "new_balance_dest": 1850000
      }' | python3 -m json.tool

echo -e "\n=== 4. Obtention d'un token JWT ==="
TOKEN=$(curl -s -X POST "$BASE_URL/auth/token" \
  -d "username=camertrust&password=changeme123" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
echo "Token obtenu (tronque) : ${TOKEN:0:20}..."

echo -e "\n=== 5. Liste des transactions (authentifie) ==="
curl -s "$BASE_URL/transactions?limit=5" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

echo -e "\n=== 6. Liste des alertes fraude (authentifie) ==="
curl -s "$BASE_URL/alerts?limit=5" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

echo -e "\n=== Demo terminee ==="
