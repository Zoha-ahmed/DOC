# Basic requirements
pip install pandas numpy scikit-learn openpyxl joblib

# For advanced features (optional)
pip install sentence-transformers  # For BERT embeddings

# Basic version
python heuristic_matcher.py

# Advanced version with ensemble methods
python advanced_heuristic_matcher.py


Think of this as an intelligent assistant for UX designers that automatically matches project requests with the right design guidelines. The system takes in two types of data: a "form guide" containing established UX heuristics (like Nielsen's 10 usability principles) and incoming project requests with details like feature descriptions, input types (buttons, dropdowns, etc.), and deadlines. Using multiple machine learning algorithms, it analyzes the text of each request, extracts key features like UI keywords and priority indicators, and then finds the best matching design heuristic from the database. The AI uses several techniques including text similarity analysis, feature matching, and ensemble learning (multiple AI models voting together) to provide recommendations with confidence scores. Finally, it generates comprehensive reports showing which UX principles apply to each project, calculates urgency levels based on deadlines, and provides actionable design recommendations - essentially automating what would normally require hours of manual review by senior designers to ensure consistent, high-quality UX guidance across all projects.RetryClaude can make mistakes. Please double-check responses.
