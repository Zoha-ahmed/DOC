import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load input files
input_df = pd.read_excel("input_data.xlsx", engine="openpyxl")
form_guide_df = pd.read_excel("form_guide_input_fields.xlsx", engine="openpyxl")

# Extract rule row and heuristic rows
rule_row = form_guide_df.iloc[0]
heuristic_rows = form_guide_df.iloc[1:]

# Map input types to standardized rules
input_type_to_rule = {
    "Dropdown": rule_row.get("Dropdown"),
    "Button": rule_row.get("Button"),
    "Button Select": rule_row.get("Button Select"),
    "Multi-Select": rule_row.get("Multi-select"),
    "Single Select": rule_row.get("Single select"),
    "Date Picker": rule_row.get("Date Picker"),
    "Date Range": rule_row.get("Date Range"),
    "Toggle": rule_row.get("Toggle"),
    "Text Input": rule_row.get("Text Input"),
    "Number Input": rule_row.get("Number Input"),
    "Duration Input": rule_row.get("Duration Input")
}

# Extract heuristics
heuristic_entries = []
for _, row in heuristic_rows.iterrows():
    stage = row["Stage"]
    for input_type in input_type_to_rule:
        detail = row.get(input_type)
        if pd.notna(detail):
            heuristic = f"{stage}: {detail}"
            heuristic_entries.append({
                "Input Type": input_type,
                "Stage": stage,
                "Heuristic": heuristic
            })

# Group heuristics by input type
heuristics_by_type = {}
for entry in heuristic_entries:
    heuristics_by_type.setdefault(entry["Input Type"], []).append(entry["Heuristic"])

# Match heuristics using TF-IDF and cosine similarity
relevant_heuristics = []
for _, row in input_df.iterrows():
    input_type = row["Input Type"]
    description = str(row["Feature Description"])
    expected_output = str(row.get("Expected Output", ""))
    context = f"{description}. Input type: {input_type}. Expected output: {expected_output}"

    heuristics = heuristics_by_type.get(input_type, [])
    if heuristics:
        corpus = [context] + heuristics
        vectorizer = TfidfVectorizer().fit(corpus)
        vectors = vectorizer.transform(corpus)
        similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
        best_index = similarities.argmax()
        relevant_heuristics.append(heuristics[best_index])
    else:
        relevant_heuristics.append("")

# Create output DataFrame
output_df = pd.DataFrame({
    "Project Name": input_df["Project Name"],
    "Input Type": input_df["Input Type"],
    "Deadline": input_df["Deadline"],
    "Stakeholder": input_df["Stakeholder"],
    "Standardized Rule": input_df["Input Type"].map(input_type_to_rule),
    "Relevant Heuristic": relevant_heuristics
})

# Save to Excel
output_df.to_excel("input_data_with_matched_heuristics_final.xlsx", index=False)
print("✅ File saved as input_data_with_matched_heuristics_final.xlsx")
