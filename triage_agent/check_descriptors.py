from database.db import DatabaseManager

db = DatabaseManager("runs/triflag.db")
rows = db.get_top_n_by_reward(3)
for row in rows:
    r = dict(row)
    print(r.get("molecule_id"), "|",
          "mol_weight:", r.get("mol_weight"), "|",
          "logp:", r.get("logp"), "|",
          "scaffold:", r.get("scaffold_smiles"))