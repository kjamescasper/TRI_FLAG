# TRI_FLAG

TRI_FLAG is a modular, agent-based triage system that automatically evaluates AI-generated small molecules prior to costly downstream workflows. The project focuses on decision-making infrastructure that improves the practical usability of existing generative chemistry tools, not on developing a new generative model.

Given candidate molecules as SMILES strings, the system performs screening based on five criteria:
•	Chemical validity - rule-based checks using RDKit
•	Physicochemical properties - MW, logP, TPSA, HBD, HBA, rotatable bonds, Lipinski Ro5 compliance
•	Synthetic accessibility - Ertl-Schuffenhauer SA score (scale 1-10); DISCARD if SA > 7
•	IP risk - Tanimoto similarity against ChEMBL (approved drugs) and SureChEMBL (patent literature)
•	Structural alerts - PAINS_A/B/C screening via RDKit FilterCatalog (advisory FLAG only)

The outcome of each evaluation is a transparent tri-flag decision: DISCARD, FLAG for review, or PASS, with an interpretable plain-English rationale. Every run is persisted to a SQLite database with a Biolink-informed schema enabling future interoperability with ChEMBL, PubChem, and the Translator Knowledge Graph ecosystem.
