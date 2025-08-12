"""
Advanced Heuristic Matching System with Multiple ML Algorithms
This enhanced version includes multiple ML approaches for better accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional: For advanced NLP (install: pip install sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Note: sentence-transformers not installed. Using TF-IDF instead of BERT embeddings.")

class AdvancedHeuristicMatcher:
    def __init__(self, form_guide_path, input_data_path, use_bert=False):
        """
        Initialize the Advanced Heuristic Matcher.
        
        Parameters:
        -----------
        form_guide_path : str
            Path to the form guide Excel file
        input_data_path : str
            Path to the input data Excel file
        use_bert : bool
            Whether to use BERT embeddings (requires sentence-transformers)
        """
        self.form_guide_path = form_guide_path
        self.input_data_path = input_data_path
        self.use_bert = use_bert and BERT_AVAILABLE
        
        # Initialize ML components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
        self.count_vectorizer = CountVectorizer(max_features=150, ngram_range=(1, 2))
        
        if self.use_bert:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ML Models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Load and prepare all data for ML processing."""
        # Load data
        self.form_guide_df = pd.read_excel(self.form_guide_path)
        self.input_df = pd.read_excel(self.input_data_path)
        
        print(f"Loaded form guide: {self.form_guide_df.shape}")
        print(f"Loaded input data: {self.input_df.shape}")
        
        # Prepare heuristics database
        self._prepare_heuristics_database()
        
    def _prepare_heuristics_database(self):
        """Transform form guide into ML-ready format."""
        input_types = [col for col in self.form_guide_df.columns if col != 'Stage']
        
        heuristics_list = []
        for _, row in self.form_guide_df.iterrows():
            stage = row['Stage']
            for input_type in input_types:
                if pd.notna(row[input_type]):
                    heuristics_list.append({
                        'Stage': stage,
                        'Input_Type': input_type,
                        'Heuristic': row[input_type],
                        'Combined_Text': f"{stage} {input_type} {row[input_type]}"
                    })
        
        self.heuristics_db = pd.DataFrame(heuristics_list)
        
        # Create label encoders
        self.label_encoders['Stage'] = LabelEncoder()
        self.label_encoders['Input_Type'] = LabelEncoder()
        
        self.heuristics_db['Stage_Encoded'] = self.label_encoders['Stage'].fit_transform(
            self.heuristics_db['Stage']
        )
        self.heuristics_db['Input_Type_Encoded'] = self.label_encoders['Input_Type'].fit_transform(
            self.heuristics_db['Input_Type']
        )
        
    def extract_advanced_features(self, text, feature_type='description'):
        """
        Extract advanced features from text using multiple techniques.
        
        Parameters:
        -----------
        text : str
            Input text to extract features from
        feature_type : str
            Type of feature extraction to use
        
        Returns:
        --------
        np.array : Feature vector
        """
        if not text or pd.isna(text):
            text = ""
        
        text = str(text).lower()
        
        features = []
        
        # Basic text statistics
        features.extend([
            len(text),  # Length
            len(text.split()),  # Word count
            text.count('.'),  # Sentence count approximation
            len([w for w in text.split() if len(w) > 6]),  # Long words
        ])
        
        # Keyword presence features
        ui_keywords = ['button', 'dropdown', 'select', 'input', 'toggle', 'filter', 
                      'display', 'show', 'hide', 'click', 'enter', 'field']
        
        for keyword in ui_keywords:
            features.append(1 if keyword in text else 0)
        
        # Priority indicators
        priority_words = ['urgent', 'critical', 'high', 'important', 'asap', 'immediate']
        features.append(sum(1 for word in priority_words if word in text))
        
        return np.array(features)
    
    def get_text_embeddings(self, texts):
        """
        Get text embeddings using BERT or TF-IDF.
        
        Parameters:
        -----------
        texts : list
            List of text strings
        
        Returns:
        --------
        np.array : Text embeddings
        """
        if self.use_bert:
            # Use BERT embeddings
            embeddings = self.bert_model.encode(texts, convert_to_numpy=True)
        else:
            # Use TF-IDF embeddings
            if len(texts) == 1:
                # For single text, fit on heuristics and transform
                all_texts = self.heuristics_db['Combined_Text'].tolist() + texts
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
                embeddings = tfidf_matrix[-1:].toarray()
            else:
                embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray()
        
        return embeddings
    
    def calculate_similarity_matrix(self, input_features, heuristic_features):
        """
        Calculate similarity between input and heuristics using multiple methods.
        
        Parameters:
        -----------
        input_features : np.array
            Features from input data
        heuristic_features : np.array
            Features from heuristics database
        
        Returns:
        --------
        np.array : Similarity scores
        """
        # Cosine similarity
        cosine_sim = cosine_similarity(input_features.reshape(1, -1), heuristic_features)[0]
        
        # Euclidean distance (converted to similarity)
        euclidean_dist = np.linalg.norm(heuristic_features - input_features, axis=1)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Manhattan distance (converted to similarity)
        manhattan_dist = np.sum(np.abs(heuristic_features - input_features), axis=1)
        manhattan_sim = 1 / (1 + manhattan_dist)
        
        # Weighted average of similarities
        final_similarity = (
            0.5 * cosine_sim + 
            0.3 * euclidean_sim + 
            0.2 * manhattan_sim
        )
        
        return final_similarity
    
    def train_ensemble_model(self):
        """
        Train an ensemble model for heuristic classification.
        This creates synthetic training data from the heuristics database.
        """
        print("\nTraining ensemble models...")
        
        # Create synthetic training data
        X_train = []
        y_train = []
        
        # Generate variations of each heuristic
        for _, row in self.heuristics_db.iterrows():
            base_text = row['Combined_Text']
            
            # Create variations
            variations = [
                base_text,
                base_text.lower(),
                base_text.upper(),
                ' '.join(base_text.split()[:5]),  # First 5 words
                ' '.join(base_text.split()[-5:]),  # Last 5 words
            ]
            
            for var in variations:
                features = self.extract_advanced_features(var)
                X_train.append(features)
                y_train.append(row['Stage_Encoded'])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            print(f"  - Trained {name}")
        
    def predict_best_heuristic(self, input_row):
        """
        Predict the best heuristic using ensemble methods.
        
        Parameters:
        -----------
        input_row : pd.Series
            Input data row
        
        Returns:
        --------
        dict : Best matching heuristic with metadata
        """
        # Combine input text
        input_text = f"{input_row.get('Feature Description', '')} {input_row.get('Expected Output', '')}"
        input_type = str(input_row.get('Input Type', ''))
        
        # Normalize input type
        input_type_normalized = self._normalize_input_type(input_type)
        
        # Extract features
        input_features = self.extract_advanced_features(input_text)
        
        # Get text embeddings
        input_embedding = self.get_text_embeddings([input_text])
        heuristic_embeddings = self.get_text_embeddings(self.heuristics_db['Combined_Text'].tolist())
        
        # Calculate similarities
        text_similarities = cosine_similarity(input_embedding, heuristic_embeddings)[0]
        
        # Filter by input type
        type_matches = self.heuristics_db['Input_Type'] == input_type_normalized
        
        # Combine scores
        final_scores = []
        for idx, row in self.heuristics_db.iterrows():
            score = text_similarities[idx]
            
            # Bonus for matching input type
            if type_matches[idx]:
                score *= 1.5
            
            # Consider priority
            priority = str(input_row.get('Priority', 'Medium')).lower()
            if priority == 'high':
                score *= 1.2
            elif priority == 'low':
                score *= 0.9
            
            final_scores.append({
                'Index': idx,
                'Stage': row['Stage'],
                'Heuristic': row['Heuristic'],
                'Input_Type': row['Input_Type'],
                'Score': score
            })
        
        # Get top match
        best_match = max(final_scores, key=lambda x: x['Score'])
        
        # Use ensemble voting if models are trained
        if hasattr(self, 'models_trained') and self.models_trained:
            input_scaled = self.scaler.transform(input_features.reshape(1, -1))
            predictions = []
            
            for name, model in self.models.items():
                pred = model.predict(input_scaled)[0]
                stage = self.label_encoders['Stage'].inverse_transform([pred])[0]
                predictions.append(stage)
            
            # Majority voting
            from collections import Counter
            most_common_stage = Counter(predictions).most_common(1)[0][0]
            
            # Filter heuristics by predicted stage
            stage_filtered = [s for s in final_scores if s['Stage'] == most_common_stage]
            if stage_filtered:
                best_match = max(stage_filtered, key=lambda x: x['Score'])
        
        return best_match
    
    def _normalize_input_type(self, input_type):
        """Normalize input type to match form guide columns."""
        mapping = {
            'dropdown': 'Dropdown',
            'toggle': 'Toggle',
            'text input': 'Text Input',
            'text': 'Text Input',
            'button': 'Button',
            'button select': 'Button Select',
            'multi-select': 'Multi-select',
            'multiselect': 'Multi-select',
            'single select': 'Single select',
            'singleselect': 'Single select',
            'date picker': 'Data Picker',
            'datepicker': 'Data Picker',
            'date range': 'Date Range',
            'daterange': 'Date Range',
            'number input': 'Number Input',
            'number': 'Number Input',
            'duration input': 'Duration Input',
            'duration': 'Duration Input'
        }
        
        input_lower = str(input_type).lower().strip()
        return mapping.get(input_lower, input_type)
    
    def get_standardized_rule(self, stage):
        """Get standardized rule description for a stage."""
        rules = {
            'Rule': 'Follow established UI/UX patterns and conventions for optimal user experience',
            'Visibility of System Status': 'Keep users informed through appropriate feedback within reasonable time',
            'Match Between System and the Real World': 'Speak the users\' language with familiar words, phrases and concepts',
            'User Control and Freedom': 'Support undo and redo, provide clearly marked emergency exits',
            'Consistency and Standards': 'Follow platform conventions, maintain consistency throughout',
            'Error Prevention': 'Eliminate error-prone conditions or check for them and present confirmation',
            'Recognition Rather Than Recall': 'Minimize memory load by making elements visible',
            'Flexibility and Efficiency of Use': 'Accelerators for expert users, flexible processes',
            'Aesthetic and Minimalist Design': 'Dialogues should not contain irrelevant information',
            'Help Users Recognize, Diagnose, and Recover from Errors': 'Express error messages in plain language',
            'Help and Documentation': 'Provide help and documentation that is easy to search'
        }
        
        return rules.get(stage, 'Apply general usability best practices')
    
    def process_all_inputs(self, use_ensemble=True):
        """
        Process all input rows using advanced ML techniques.
        
        Parameters:
        -----------
        use_ensemble : bool
            Whether to use ensemble methods
        
        Returns:
        --------
        pd.DataFrame : Results with recommendations
        """
        results = []
        
        # Train ensemble if requested
        if use_ensemble:
            self.train_ensemble_model()
            self.models_trained = True
        else:
            self.models_trained = False
        
        print("\nProcessing inputs...")
        
        for idx, row in self.input_df.iterrows():
            # Get best heuristic
            best_match = self.predict_best_heuristic(row)
            
            # Get standardized rule
            std_rule = self.get_standardized_rule(best_match['Stage'])
            
            # Calculate deadline urgency
            urgency = 'Normal'
            if 'Deadline' in row and pd.notna(row['Deadline']):
                try:
                    deadline = pd.to_datetime(row['Deadline'])
                    days_until = (deadline - datetime.now()).days
                    if days_until < 7:
                        urgency = 'Critical'
                    elif days_until < 14:
                        urgency = 'High'
                    elif days_until < 30:
                        urgency = 'Medium'
                except:
                    pass
            
            result = {
                'Project Name': row.get('Project Name', ''),
                'Input Type': row.get('Input Type', ''),
                'Deadline': row.get('Deadline', ''),
                'Stakeholder': row.get('Stakeholder', ''),
                'Standardized Rule': std_rule,
                'Relevant Heuristic': best_match['Heuristic'],
                'Heuristic Category': best_match['Stage'],
                'Matched Input Type': best_match['Input_Type'],
                'Confidence Score': f"{best_match['Score']:.2%}",
                'Urgency Level': urgency,
                'Feature Description': row.get('Feature Description', ''),
                'Priority': row.get('Priority', '')
            }
            
            results.append(result)
            print(f"  - Processed: {row.get('Project Name', 'Unknown')}")
        
        return pd.DataFrame(results)
    
    def generate_insights(self, results_df):
        """
        Generate insights from the processed results.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Processed results dataframe
        
        Returns:
        --------
        dict : Insights and statistics
        """
        insights = {
            'total_projects': len(results_df),
            'unique_stakeholders': results_df['Stakeholder'].nunique(),
            'most_common_input_type': results_df['Input Type'].mode()[0] if not results_df.empty else 'N/A',
            'heuristic_distribution': results_df['Heuristic Category'].value_counts().to_dict(),
            'urgency_distribution': results_df['Urgency Level'].value_counts().to_dict(),
            'average_confidence': results_df['Confidence Score'].str.rstrip('%').astype(float).mean()
        }
        
        # High priority items
        high_priority = results_df[results_df['Priority'] == 'High']
        insights['high_priority_count'] = len(high_priority)
        
        # Critical deadlines
        critical = results_df[results_df['Urgency Level'] == 'Critical']
        insights['critical_items'] = critical[['Project Name', 'Deadline']].to_dict('records')
        
        return insights
    
    def save_comprehensive_results(self, output_path='advanced_heuristic_results.xlsx'):
        """
        Save comprehensive results with multiple sheets.
        
        Parameters:
        -----------
        output_path : str
            Output Excel file path
        """
        # Process data
        self.load_and_prepare_data()
        results_df = self.process_all_inputs(use_ensemble=True)
        
        # Generate insights
        insights = self.generate_insights(results_df)
        
        # Create insights dataframe
        insights_df = pd.DataFrame([
            {'Metric': 'Total Projects', 'Value': insights['total_projects']},
            {'Metric': 'Unique Stakeholders', 'Value': insights['unique_stakeholders']},
            {'Metric': 'Most Common Input Type', 'Value': insights['most_common_input_type']},
            {'Metric': 'High Priority Items', 'Value': insights['high_priority_count']},
            {'Metric': 'Average Confidence Score', 'Value': f"{insights['average_confidence']:.1f}%"},
        ])
        
        # Heuristic distribution
        heuristic_dist_df = pd.DataFrame(
            list(insights['heuristic_distribution'].items()),
            columns=['Heuristic Category', 'Count']
        )
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Insights
            insights_df.to_excel(writer, sheet_name='Summary Insights', index=False)
            
            # Heuristic distribution
            heuristic_dist_df.to_excel(writer, sheet_name='Heuristic Distribution', index=False)
            
            # High priority items
            high_priority_df = results_df[results_df['Priority'] == 'High']
            if not high_priority_df.empty:
                high_priority_df.to_excel(writer, sheet_name='High Priority Items', index=False)
            
            # Format sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\nResults saved to: {output_path}")
        print(f"Total recommendations: {len(results_df)}")
        print(f"Sheets created: Recommendations, Summary Insights, Heuristic Distribution")
        
        return results_df, insights


def main():
    """Main execution function for advanced heuristic matching."""
    # Configuration
    form_guide_path = 'form_guide_input_fields.xlsx'
    input_data_path = 'input_data 1.xlsx'
    output_path = 'advanced_heuristic_results.xlsx'
    
    print("=" * 70)
    print("ADVANCED HEURISTIC MATCHING SYSTEM WITH ML")
    print("=" * 70)
    
    try:
        # Initialize advanced matcher
        matcher = AdvancedHeuristicMatcher(
            form_guide_path, 
            input_data_path,
            use_bert=False  # Set to True if sentence-transformers is installed
        )
        
        # Process and save results
        results, insights = matcher.save_comprehensive_results(output_path)
        
        # Display summary
        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Projects Processed: {insights['total_projects']}")
        print(f"Unique Stakeholders: {insights['unique_stakeholders']}")
        print(f"Most Common Input Type: {insights['most_common_input_type']}")
        print(f"High Priority Items: {insights['high_priority_count']}")
        print(f"Average Confidence: {insights['average_confidence']:.1f}%")
        
        if insights['critical_items']:
            print("\n⚠️  CRITICAL DEADLINES:")
            for item in insights['critical_items']:
                print(f"  - {item['Project Name']}: {item['Deadline']}")
        
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE!")
        print(f"📁 Output saved to: {output_path}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check that input files exist and have the correct format.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
