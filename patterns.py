import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class UIComponentOptimizer:
    def __init__(self, input_data_path, form_guide_path):
        """
        Initialize the optimizer with file paths
        
        Args:
            input_data_path: Path to input_data.xlsx
            form_guide_path: Path to form_guide_input_fields.xlsx
        """
        self.input_data = pd.read_excel(input_data_path)
        self.form_guide = pd.read_excel(form_guide_path)
        self.heuristics = self._extract_heuristics()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    def _extract_heuristics(self):
        """Extract heuristics from form guide (rows represent different heuristic stages)"""
        heuristics = {
            'Visibility of System Status': 1,
            'Match Between System and the Real World': 2,
            'User Control and Freedom': 3,
            'Consistency & Standards': 4,
            'Error Prevention': 5,
            'Recognition Rather than Recall': 6,
            'Flexibility & Efficiency of Use': 7,
            'Aesthetic and Minimalistic Design': 8,
            'Help Users Recognize, Diagnose, and Recover from Errors': 9
        }
        return heuristics
    
    def _get_standardized_rule(self, input_type):
        """Get the standardized rule for a given input type from form guide"""
        # Create a case-insensitive mapping of column names
        column_mapping = {}
        for col in self.form_guide.columns:
            if col != 'Stage':  # Skip the Stage column
                # Store both original and lowercase versions
                column_mapping[col.lower().replace(' ', '')] = col
                column_mapping[col.lower()] = col
                column_mapping[col] = col
        
        # Try different variations of the input type
        variations = [
            input_type,  # Original
            input_type.lower(),  # Lowercase
            input_type.lower().replace(' ', ''),  # Lowercase no spaces
            input_type.replace(' ', ''),  # No spaces
        ]
        
        for variant in variations:
            if variant in column_mapping:
                actual_column = column_mapping[variant]
                # Get the rule (first row contains the standardized rules)
                # The first row (index 0) has Stage='Rule' and contains the rules
                return self.form_guide[actual_column].iloc[0]
        
        return f"No standardized rule found for '{input_type}'"
    
    def _calculate_heuristic_scores(self, feature_desc, input_type, expected_output):
        """
        Calculate relevance scores for each heuristic using ML techniques
        
        Args:
            feature_desc: Feature description text
            input_type: Type of input component
            expected_output: Expected output description
        
        Returns:
            Dictionary of heuristic scores
        """
        scores = {}
        
        # Combine all text features
        combined_text = f"{feature_desc} {expected_output}"
        
        # Create case-insensitive column mapping
        column_mapping = {}
        for col in self.form_guide.columns:
            if col != 'Stage':
                column_mapping[col.lower().replace(' ', '')] = col
                column_mapping[col.lower()] = col
                column_mapping[col] = col
        
        # Find the actual column name
        actual_column = None
        variations = [
            input_type,
            input_type.lower(),
            input_type.lower().replace(' ', ''),
            input_type.replace(' ', ''),
        ]
        
        for variant in variations:
            if variant in column_mapping:
                actual_column = column_mapping[variant]
                break
        
        # Get relevant heuristic descriptions for the input type
        if actual_column:
            heuristic_descriptions = self.form_guide[actual_column].iloc[1:].tolist()
            
            # Calculate similarity scores using TF-IDF and cosine similarity
            all_texts = [combined_text] + heuristic_descriptions
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between input and each heuristic
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # Map similarities to heuristic names
            heuristic_names = list(self.heuristics.keys())
            for i, (heuristic, sim_score) in enumerate(zip(heuristic_names, similarities)):
                scores[heuristic] = sim_score
        else:
            # If no matching column found, use default scores
            for heuristic in self.heuristics.keys():
                scores[heuristic] = 0.1
                
        # Apply keyword-based boosting for better accuracy
        scores = self._apply_keyword_boosting(feature_desc, expected_output, scores)
        
        return scores
    
    def _apply_keyword_boosting(self, feature_desc, expected_output, scores):
        """Apply keyword-based rules to boost relevant heuristic scores"""
        
        text_lower = (feature_desc + " " + expected_output).lower()
        
        # Keyword mappings to heuristics
        keyword_rules = {
            'Visibility of System Status': ['status', 'feedback', 'indicator', 'show', 'display', 'highlight'],
            'Match Between System and the Real World': ['familiar', 'real', 'natural', 'language', 'terms'],
            'User Control and Freedom': ['undo', 'cancel', 'back', 'reset', 'clear', 'change'],
            'Consistency & Standards': ['consistent', 'standard', 'uniform', 'same', 'similar'],
            'Error Prevention': ['prevent', 'validate', 'check', 'avoid', 'error', 'mistake'],
            'Recognition Rather than Recall': ['recognize', 'visible', 'clear', 'obvious', 'prominent'],
            'Flexibility & Efficiency of Use': ['shortcut', 'quick', 'efficient', 'bulk', 'fast'],
            'Aesthetic and Minimalistic Design': ['simple', 'clean', 'minimal', 'aesthetic', 'design'],
            'Help Users Recognize, Diagnose, and Recover from Errors': ['error', 'help', 'message', 'guide', 'recover']
        }
        
        # Boost scores based on keyword presence
        for heuristic, keywords in keyword_rules.items():
            if heuristic in scores:
                for keyword in keywords:
                    if keyword in text_lower:
                        scores[heuristic] *= 1.3  # Boost by 30%
        
        return scores
    
    def _select_optimal_heuristic(self, scores, priority='High'):
        """
        Select the most relevant heuristic based on scores and priority
        
        Args:
            scores: Dictionary of heuristic scores
            priority: Priority level of the feature
        
        Returns:
            Optimal heuristic name
        """
        if not scores:
            return "Consistency & Standards"  # Default heuristic
        
        # Apply priority weighting
        priority_weights = {'High': 1.2, 'Medium': 1.0, 'Low': 0.8}
        weight = priority_weights.get(priority, 1.0)
        
        # Weight scores based on priority
        weighted_scores = {k: v * weight for k, v in scores.items()}
        
        # Select heuristic with highest score
        optimal_heuristic = max(weighted_scores, key=weighted_scores.get)
        
        return optimal_heuristic
    
    def process_data(self):
        """
        Process all input data and generate recommendations
        
        Returns:
            DataFrame with recommendations
        """
        results = []
        
        for idx, row in self.input_data.iterrows():
            # Extract input data
            project_name = row['Project Name']
            feature_desc = row['Feature Description']
            input_type = row['Input Type']
            expected_output = row['Expected Output']
            deadline = row['Deadline']
            stakeholder = row['Stakeholder']
            priority = row.get('Priority', 'Medium')
            
            # Get standardized rule
            standardized_rule = self._get_standardized_rule(input_type)
            
            # Calculate heuristic scores using ML
            heuristic_scores = self._calculate_heuristic_scores(
                feature_desc, input_type, expected_output
            )
            
            # Select optimal heuristic
            optimal_heuristic = self._select_optimal_heuristic(heuristic_scores, priority)
            
            # Get the specific heuristic guideline for this input type
            heuristic_guideline = ""
            
            # Create case-insensitive column mapping
            column_mapping = {}
            for col in self.form_guide.columns:
                if col != 'Stage':
                    column_mapping[col.lower().replace(' ', '')] = col
                    column_mapping[col.lower()] = col
                    column_mapping[col] = col
            
            # Find the actual column name
            actual_column = None
            variations = [
                input_type,
                input_type.lower(),
                input_type.lower().replace(' ', ''),
                input_type.replace(' ', ''),
            ]
            
            for variant in variations:
                if variant in column_mapping:
                    actual_column = column_mapping[variant]
                    break
            
            if actual_column:
                heuristic_idx = self.heuristics.get(optimal_heuristic, 1)
                if heuristic_idx < len(self.form_guide):
                    heuristic_guideline = self.form_guide[actual_column].iloc[heuristic_idx]
            
            # Compile result
            result = {
                'Project Name': project_name,
                'Input Type': input_type,
                'Deadline': deadline,
                'Stakeholder': stakeholder,
                'Standardized Rule': standardized_rule,
                'Relevant Heuristic': optimal_heuristic,
                'Heuristic Guideline': heuristic_guideline,
                'Confidence Score': round(max(heuristic_scores.values()) * 100, 2) if heuristic_scores else 0
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_report(self, output_path='ui_optimization_report.xlsx'):
        """
        Generate and save the optimization report
        
        Args:
            output_path: Path to save the output Excel file
        """
        # Process data
        results_df = self.process_data()
        
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results sheet
            results_df.to_excel(writer, sheet_name='Optimization Results', index=False)
            
            # Summary statistics sheet
            summary_stats = self._generate_summary_stats(results_df)
            summary_stats.to_excel(writer, sheet_name='Summary Statistics')
            
            # Heuristic frequency sheet
            heuristic_freq = results_df['Relevant Heuristic'].value_counts().to_frame()
            heuristic_freq.columns = ['Frequency']
            heuristic_freq.to_excel(writer, sheet_name='Heuristic Distribution')
        
        print(f"✅ Report generated successfully: {output_path}")
        return results_df
    
    def _generate_summary_stats(self, results_df):
        """Generate summary statistics for the report"""
        stats = {
            'Total Projects': len(results_df['Project Name'].unique()),
            'Total Features': len(results_df),
            'Average Confidence Score': results_df['Confidence Score'].mean(),
            'Most Common Input Type': results_df['Input Type'].mode()[0] if not results_df.empty else 'N/A',
            'Most Applied Heuristic': results_df['Relevant Heuristic'].mode()[0] if not results_df.empty else 'N/A'
        }
        return pd.Series(stats)
    
    def display_results(self, results_df=None):
        """Display results in a formatted way"""
        if results_df is None:
            results_df = self.process_data()
        
        print("\n" + "="*80)
        print("UI COMPONENT OPTIMIZATION RESULTS")
        print("="*80)
        
        for idx, row in results_df.iterrows():
            print(f"\n📋 Project: {row['Project Name']}")
            print(f"   Input Type: {row['Input Type']}")
            print(f"   Deadline: {row['Deadline']}")
            print(f"   Stakeholder: {row['Stakeholder']}")
            print(f"   📐 Standardized Rule: {row['Standardized Rule']}")
            print(f"   🎯 Optimal Heuristic: {row['Relevant Heuristic']}")
            print(f"   📝 Heuristic Guideline: {row['Heuristic Guideline']}")
            print(f"   🔍 Confidence Score: {row['Confidence Score']}%")
            print("-"*80)
    
    def debug_column_matching(self):
        """Debug function to show column name matching"""
        print("\n🔍 DEBUG: Column Matching Information")
        print("="*60)
        print("Form Guide Columns (excluding 'Stage'):")
        for col in self.form_guide.columns:
            if col != 'Stage':
                print(f"  - '{col}'")
        
        print("\nInput Data 'Input Type' values:")
        for input_type in self.input_data['Input Type'].unique():
            print(f"  - '{input_type}'")
            # Test the matching
            rule = self._get_standardized_rule(input_type)
            print(f"    → Rule: {rule[:50]}..." if len(rule) > 50 else f"    → Rule: {rule}")
        print("="*60)


# Main execution function
def main():
    """Main function to run the UI optimization process"""
    
    # File paths
    input_data_path = 'input_data.xlsx'
    form_guide_path = 'form_guide_input_fields.xlsx'
    output_path = 'ui_optimization_report.xlsx'
    
    try:
        # Initialize optimizer
        print("🚀 Initializing UI Component Optimizer...")
        optimizer = UIComponentOptimizer(input_data_path, form_guide_path)
        
        # Generate report
        print("📊 Processing data and applying ML techniques...")
        results = optimizer.generate_report(output_path)
        
        # Display results
        optimizer.display_results(results)
        
        print(f"\n✨ Optimization complete! Results saved to '{output_path}'")
        
        # Display summary statistics
        print("\n📈 SUMMARY STATISTICS:")
        print("-"*40)
        summary = optimizer._generate_summary_stats(results)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found. Please ensure both input files exist:")
        print(f"   - {input_data_path}")
        print(f"   - {form_guide_path}")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()