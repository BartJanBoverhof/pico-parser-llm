import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from openpyxl import load_workbook


class ResultsAnalyzer:
    """
    Analyzes validated PICO extraction results from RAG-LLM pipeline.
    
    Handles HPO (hyperparameter optimization) and consistency check results
    for multiple disease cases (NSCLC, HCC).
    """
    
    def __init__(self, results_folder: str = "results/scored"):
        """
        Initialize the results analyzer.
        
        Args:
            results_folder: Path to folder containing scored .xlsx files
        """
        self.results_folder = Path(results_folder)
        self.cases = ["NSCLC", "HCC"]
        self.analysis_types = ["hpo", "con"]
        
        # Different scenario names for HPO vs consistency checks
        self.scenarios_hpo = ["base", "hpo1", "hpo2", "hpo3", "hpo4", "hpo5", "hpo6"]
        self.scenarios_con = ["base", "base_b", "base_c", "base_d", "base_e"]
        
        self.pico_elements = ["Population", "Comparator", "Outcome"]
        
        # Store results
        self.results = {}
        
    def read_excel_file(self, filepath: Path, case: str, scenarios: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Read an Excel file and extract relevant sheets using openpyxl directly.
        
        Args:
            filepath: Path to Excel file
            case: Case name (e.g., 'NSCLC', 'HCC')
            scenarios: List of scenario names to look for
            
        Returns:
            Dictionary with scenario names as keys and DataFrames as values
        """
        # Load workbook with openpyxl directly
        wb = load_workbook(filepath, read_only=True, data_only=True)
        available_sheets = wb.sheetnames
        
        data = {}
        
        for scenario in scenarios:
            # Read PC (Population & Comparator) sheet
            pc_sheet_name = f"{case}_PC_{scenario}"
            if pc_sheet_name in available_sheets:
                ws = wb[pc_sheet_name]
                # Convert worksheet to DataFrame
                data_rows = []
                headers = None
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i == 0:
                        headers = row
                    else:
                        data_rows.append(row)
                df_pc = pd.DataFrame(data_rows, columns=headers)
                data[f"{scenario}_PC"] = df_pc
            
            # Read O (Outcome) sheet
            o_sheet_name = f"{case}_O_{scenario}"
            if o_sheet_name in available_sheets:
                ws = wb[o_sheet_name]
                # Convert worksheet to DataFrame
                data_rows = []
                headers = None
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i == 0:
                        headers = row
                    else:
                        data_rows.append(row)
                df_o = pd.DataFrame(data_rows, columns=headers)
                data[f"{scenario}_O"] = df_o
        
        wb.close()
        return data
    
    def extract_scores(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, List[float]]:
        """
        Extract score rows from a DataFrame.
        
        Args:
            df: DataFrame containing validation scores
            columns: List of column names to extract (e.g., ['Population', 'Comparator'])
            
        Returns:
            Dictionary with column names as keys and lists of scores as values
        """
        # Filter rows ending with '_score'
        score_rows = df[df.iloc[:, 0].astype(str).str.endswith('_score', na=False)]
        
        scores = {}
        for col in columns:
            if col in score_rows.columns:
                # Extract scores, convert to float, and replace NaN/empty with 0
                col_scores = pd.to_numeric(score_rows[col], errors='coerce').fillna(0).tolist()
                scores[col] = col_scores
            else:
                scores[col] = []
        
        return scores
    
    def calculate_recall(self, scores: List[float]) -> float:
        """
        Calculate recall score (average of all scores).
        
        Args:
            scores: List of individual scores (0-1)
            
        Returns:
            Average recall score
        """
        if not scores:
            return 0.0
        return np.mean(scores)
    
    def analyze_case(self, case: str, analysis_type: str) -> Dict:
        """
        Analyze results for a specific case and analysis type.
        
        Args:
            case: Case name (e.g., 'NSCLC', 'HCC')
            analysis_type: 'hpo' or 'con'
            
        Returns:
            Dictionary containing analysis results
        """
        # Construct filename
        filename = f"{case}_{analysis_type}.xlsx"
        filepath = self.results_folder / filename
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return {}
        
        print(f"\nAnalyzing {case} - {analysis_type.upper()}...")
        
        # Select appropriate scenario list
        scenarios = self.scenarios_hpo if analysis_type == "hpo" else self.scenarios_con
        
        # Read Excel file
        sheets_data = self.read_excel_file(filepath, case, scenarios)
        
        # Store results for each scenario
        scenario_results = {}
        
        for scenario in scenarios:
            scenario_scores = {
                'Population': [],
                'Comparator': [],
                'Outcome': []
            }
            
            # Extract PC scores
            pc_key = f"{scenario}_PC"
            if pc_key in sheets_data:
                pc_scores = self.extract_scores(
                    sheets_data[pc_key], 
                    ['Population', 'Comparator']
                )
                scenario_scores['Population'] = pc_scores.get('Population', [])
                scenario_scores['Comparator'] = pc_scores.get('Comparator', [])
            
            # Extract O scores
            o_key = f"{scenario}_O"
            if o_key in sheets_data:
                o_scores = self.extract_scores(
                    sheets_data[o_key], 
                    ['Outcome']
                )
                scenario_scores['Outcome'] = o_scores.get('Outcome', [])
            
            # Calculate recall for each PICO element
            scenario_results[scenario] = {
                'Population': {
                    'scores': scenario_scores['Population'],
                    'recall': self.calculate_recall(scenario_scores['Population']),
                    'n': len(scenario_scores['Population'])
                },
                'Comparator': {
                    'scores': scenario_scores['Comparator'],
                    'recall': self.calculate_recall(scenario_scores['Comparator']),
                    'n': len(scenario_scores['Comparator'])
                },
                'Outcome': {
                    'scores': scenario_scores['Outcome'],
                    'recall': self.calculate_recall(scenario_scores['Outcome']),
                    'n': len(scenario_scores['Outcome'])
                }
            }
            
            # Calculate total recall (average of P, C, O)
            recalls = [
                scenario_results[scenario]['Population']['recall'],
                scenario_results[scenario]['Comparator']['recall'],
                scenario_results[scenario]['Outcome']['recall']
            ]
            scenario_results[scenario]['Total'] = {
                'recall': np.mean(recalls),
                'n': sum([
                    scenario_results[scenario]['Population']['n'],
                    scenario_results[scenario]['Comparator']['n'],
                    scenario_results[scenario]['Outcome']['n']
                ])
            }
        
        return scenario_results
    
    def run_analysis(self):
        """
        Run analysis for all cases and analysis types.
        """
        for case in self.cases:
            for analysis_type in self.analysis_types:
                key = f"{case}_{analysis_type}"
                self.results[key] = self.analyze_case(case, analysis_type)
    
    def print_results(self):
        """
        Print analysis results in a clear, formatted manner.
        """
        print("\n" + "="*80)
        print("RAG-LLM PIPELINE RESULTS ANALYSIS")
        print("="*80)
        
        for case in self.cases:
            for analysis_type in self.analysis_types:
                key = f"{case}_{analysis_type}"
                
                if key not in self.results or not self.results[key]:
                    continue
                
                print(f"\n{'─'*80}")
                print(f"CASE: {case} | TYPE: {analysis_type.upper()}")
                print(f"{'─'*80}")
                
                results = self.results[key]
                
                # Select appropriate scenario list
                scenarios = self.scenarios_hpo if analysis_type == "hpo" else self.scenarios_con
                
                # Print header
                print(f"\n{'Scenario':<12} {'Total':>8} {'Population':>12} {'Comparator':>12} {'Outcome':>10} {'N':>6}")
                print("─" * 80)
                
                # Print results for each scenario
                for scenario in scenarios:
                    if scenario not in results:
                        continue
                    
                    r = results[scenario]
                    print(f"{scenario:<12} "
                          f"{r['Total']['recall']:>8.3f} "
                          f"{r['Population']['recall']:>12.3f} "
                          f"{r['Comparator']['recall']:>12.3f} "
                          f"{r['Outcome']['recall']:>10.3f} "
                          f"{r['Total']['n']:>6}")
                
                # Calculate and print summary statistics
                print("\n" + "─" * 80)
                print("SUMMARY STATISTICS")
                print("─" * 80)
                
                # Average across scenarios
                total_recalls = [results[s]['Total']['recall'] for s in scenarios if s in results]
                pop_recalls = [results[s]['Population']['recall'] for s in scenarios if s in results]
                comp_recalls = [results[s]['Comparator']['recall'] for s in scenarios if s in results]
                out_recalls = [results[s]['Outcome']['recall'] for s in scenarios if s in results]
                
                print(f"{'Average':<12} "
                      f"{np.mean(total_recalls):>8.3f} "
                      f"{np.mean(pop_recalls):>12.3f} "
                      f"{np.mean(comp_recalls):>12.3f} "
                      f"{np.mean(out_recalls):>10.3f}")
                
                print(f"{'Std Dev':<12} "
                      f"{np.std(total_recalls):>8.3f} "
                      f"{np.std(pop_recalls):>12.3f} "
                      f"{np.std(comp_recalls):>12.3f} "
                      f"{np.std(out_recalls):>10.3f}")
                
                print(f"{'Min':<12} "
                      f"{np.min(total_recalls):>8.3f} "
                      f"{np.min(pop_recalls):>12.3f} "
                      f"{np.min(comp_recalls):>12.3f} "
                      f"{np.min(out_recalls):>10.3f}")
                
                print(f"{'Max':<12} "
                      f"{np.max(total_recalls):>8.3f} "
                      f"{np.max(pop_recalls):>12.3f} "
                      f"{np.max(comp_recalls):>12.3f} "
                      f"{np.max(out_recalls):>10.3f}")
    
    
    def get_best_scenario(self, case: str, analysis_type: str) -> Tuple[str, float]:
        """
        Get the best performing scenario for a given case and analysis type.
        
        Args:
            case: Case name
            analysis_type: 'hpo' or 'con'
            
        Returns:
            Tuple of (scenario_name, total_recall)
        """
        key = f"{case}_{analysis_type}"
        
        if key not in self.results or not self.results[key]:
            return None, 0.0
        
        results = self.results[key]
        
        # Select appropriate scenario list
        scenarios = self.scenarios_hpo if analysis_type == "hpo" else self.scenarios_con
        
        best_scenario = None
        best_recall = 0.0
        
        for scenario in scenarios:
            if scenario not in results:
                continue
            
            recall = results[scenario]['Total']['recall']
            if recall > best_recall:
                best_recall = recall
                best_scenario = scenario
        
        return best_scenario, best_recall


