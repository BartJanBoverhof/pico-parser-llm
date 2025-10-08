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
    Calculates recall, precision, and F1 scores.
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
        # Store precision data separately
        self.precision_data = {}
        
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
    
    def load_precision_data(self, case: str):
        """
        Load precision data from separate precision files.
        
        Args:
            case: Case name (e.g., 'NSCLC', 'HCC')
        """
        # Construct precision filename
        precision_filename = f"{case}_precision.xlsx"
        precision_filepath = self.results_folder / precision_filename
        
        if not precision_filepath.exists():
            print(f"Warning: Precision file not found: {precision_filepath}")
            return
        
        print(f"Loading precision data for {case}...")
        print(f"  File path: {precision_filepath}")
        print(f"  File exists: {precision_filepath.exists()}")
        print(f"  File size: {precision_filepath.stat().st_size} bytes")
        
        # First, check what sheets actually exist in the file
        try:
            wb = load_workbook(precision_filepath, read_only=True, data_only=True)
            actual_sheets = wb.sheetnames
            print(f"  Actual sheets in file: {actual_sheets}")
            wb.close()
        except Exception as e:
            print(f"  ✗ Error reading workbook: {type(e).__name__}: {e}")
            return
        
        if not actual_sheets:
            print(f"  ✗ No sheets found in file. File may be corrupted or in wrong format.")
            print(f"  Try opening the file in Excel to verify it's valid.")
            return
        
        # Read only the 'base' scenario from precision file
        precision_sheets = self.read_excel_file(precision_filepath, case, ["base"])
        
        print(f"  Looking for sheet: {case}_PC_base")
        print(f"  Sheets loaded: {list(precision_sheets.keys())}")
        
        if "base_PC" in precision_sheets:
            self.precision_data[case] = precision_sheets["base_PC"]
            print(f"  ✓ Successfully loaded {len(precision_sheets['base_PC'])} rows")
            print(f"  ✓ Columns: {list(precision_sheets['base_PC'].columns)}")
            
            # Extract and preview precision scores
            test_scores = self.extract_scores(precision_sheets["base_PC"], ['Population', 'Comparator'])
            print(f"  ✓ Found {len(test_scores.get('Population', []))} Population scores")
            print(f"  ✓ Found {len(test_scores.get('Comparator', []))} Comparator scores")
            if test_scores.get('Population'):
                print(f"  ✓ Sample Population scores: {test_scores['Population'][:3]}")
            if test_scores.get('Comparator'):
                print(f"  ✓ Sample Comparator scores: {test_scores['Comparator'][:3]}")
        else:
            print(f"  ✗ Sheet 'base_PC' not found in loaded data")
            print(f"  Available keys: {list(precision_sheets.keys())}")
    
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
    
    def calculate_precision(self, scores: List[float]) -> float:
        """
        Calculate precision score (average of all scores).
        
        Args:
            scores: List of individual scores (0-1)
            
        Returns:
            Average precision score
        """
        if not scores:
            return 0.0
        return np.mean(scores)
    
    def calculate_f1(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def analyze_case(self, case: str, analysis_type: str) -> Dict:
        """
        Analyze results for a specific case and analysis type.
        
        Args:
            case: Case name (e.g., 'NSCLC', 'HCC')
            analysis_type: 'hpo' or 'con'
            
        Returns:
            Dictionary containing analysis results
        """
        # Construct filename for recall data
        filename = f"{case}_{analysis_type}.xlsx"
        filepath = self.results_folder / filename
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return {}
        
        print(f"\nAnalyzing {case} - {analysis_type.upper()}...")
        
        # Select appropriate scenario list
        scenarios = self.scenarios_hpo if analysis_type == "hpo" else self.scenarios_con
        
        # Read Excel file (recall data)
        sheets_data = self.read_excel_file(filepath, case, scenarios)
        
        # Store results for each scenario
        scenario_results = {}
        
        for scenario in scenarios:
            scenario_scores = {
                'Population': {'recall': [], 'precision': []},
                'Comparator': {'recall': [], 'precision': []},
                'Outcome': {'recall': []}
            }
            
            # Extract PC recall scores
            pc_key = f"{scenario}_PC"
            if pc_key in sheets_data:
                pc_recall_scores = self.extract_scores(
                    sheets_data[pc_key], 
                    ['Population', 'Comparator']
                )
                scenario_scores['Population']['recall'] = pc_recall_scores.get('Population', [])
                scenario_scores['Comparator']['recall'] = pc_recall_scores.get('Comparator', [])
            
            # Extract O recall scores (no precision for Outcome)
            o_key = f"{scenario}_O"
            if o_key in sheets_data:
                o_recall_scores = self.extract_scores(
                    sheets_data[o_key], 
                    ['Outcome']
                )
                scenario_scores['Outcome']['recall'] = o_recall_scores.get('Outcome', [])
            
            # Extract precision scores (only for 'base' scenario from separate file)
            if scenario == 'base' and case in self.precision_data:
                pc_precision_scores = self.extract_scores(
                    self.precision_data[case],
                    ['Population', 'Comparator']
                )
                scenario_scores['Population']['precision'] = pc_precision_scores.get('Population', [])
                scenario_scores['Comparator']['precision'] = pc_precision_scores.get('Comparator', [])
            
            # Calculate metrics for each PICO element
            scenario_results[scenario] = {}
            
            # Population
            pop_recall = self.calculate_recall(scenario_scores['Population']['recall'])
            pop_precision = self.calculate_precision(scenario_scores['Population']['precision']) if scenario == 'base' and scenario_scores['Population']['precision'] else None
            pop_f1 = self.calculate_f1(pop_precision, pop_recall) if pop_precision is not None else None
            
            scenario_results[scenario]['Population'] = {
                'recall_scores': scenario_scores['Population']['recall'],
                'recall': pop_recall,
                'precision': pop_precision,
                'f1': pop_f1,
                'n': len(scenario_scores['Population']['recall'])
            }
            
            # Comparator
            comp_recall = self.calculate_recall(scenario_scores['Comparator']['recall'])
            comp_precision = self.calculate_precision(scenario_scores['Comparator']['precision']) if scenario == 'base' and scenario_scores['Comparator']['precision'] else None
            comp_f1 = self.calculate_f1(comp_precision, comp_recall) if comp_precision is not None else None
            
            scenario_results[scenario]['Comparator'] = {
                'recall_scores': scenario_scores['Comparator']['recall'],
                'recall': comp_recall,
                'precision': comp_precision,
                'f1': comp_f1,
                'n': len(scenario_scores['Comparator']['recall'])
            }
            
            # Outcome (no precision available)
            out_recall = self.calculate_recall(scenario_scores['Outcome']['recall'])
            
            scenario_results[scenario]['Outcome'] = {
                'recall_scores': scenario_scores['Outcome']['recall'],
                'recall': out_recall,
                'precision': None,
                'f1': None,
                'n': len(scenario_scores['Outcome']['recall'])
            }
            
            # Calculate total metrics (average of P, C, O)
            total_recall = np.mean([pop_recall, comp_recall, out_recall])
            
            # For precision and F1, only average P and C (O doesn't have precision)
            if scenario == 'base' and pop_precision is not None and comp_precision is not None:
                total_precision = np.mean([pop_precision, comp_precision])
                total_f1 = self.calculate_f1(total_precision, total_recall)
            else:
                total_precision = None
                total_f1 = None
            
            scenario_results[scenario]['Total'] = {
                'recall': total_recall,
                'precision': total_precision,
                'f1': total_f1,
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
        # First, load precision data for all cases
        for case in self.cases:
            self.load_precision_data(case)
        
        # Then analyze recall and combine with precision
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
                
                # Print RECALL results
                print(f"\n{'RECALL SCORES':^80}")
                print(f"{'Scenario':<12} {'Total':>8} {'Population':>12} {'Comparator':>12} {'Outcome':>10} {'N':>6}")
                print("─" * 80)
                
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
                
                # Calculate and print summary statistics for RECALL
                print("\n" + "─" * 80)
                print("RECALL SUMMARY STATISTICS")
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
        
        # Print PRECISION and F1 scores separately (once per case)
        print("\n" + "="*80)
        print("PRECISION & F1 SCORES (BASE SCENARIO)")
        print("="*80)
        
        for case in self.cases:
            # Check if precision data exists for this case
            has_precision = False
            base_metrics = None
            
            # Check both HPO and CON to find base scenario with precision
            for analysis_type in self.analysis_types:
                key = f"{case}_{analysis_type}"
                if key in self.results and 'base' in self.results[key]:
                    if self.results[key]['base']['Total']['precision'] is not None:
                        has_precision = True
                        base_metrics = self.results[key]['base']
                        break
            
            if has_precision and base_metrics:
                print(f"\n{'─'*80}")
                print(f"CASE: {case}")
                print(f"{'─'*80}")
                
                print(f"\n{'Metric':<15} {'Total':>8} {'Population':>12} {'Comparator':>12} {'Outcome':>10}")
                print("─" * 80)
                
                # Recall
                print(f"{'Recall':<15} "
                      f"{base_metrics['Total']['recall']:>8.3f} "
                      f"{base_metrics['Population']['recall']:>12.3f} "
                      f"{base_metrics['Comparator']['recall']:>12.3f} "
                      f"{base_metrics['Outcome']['recall']:>10.3f}")
                
                # Precision
                print(f"{'Precision':<15} "
                      f"{base_metrics['Total']['precision']:>8.3f} "
                      f"{base_metrics['Population']['precision']:>12.3f} "
                      f"{base_metrics['Comparator']['precision']:>12.3f} "
                      f"{'N/A':>10}")
                
                # F1
                print(f"{'F1 Score':<15} "
                      f"{base_metrics['Total']['f1']:>8.3f} "
                      f"{base_metrics['Population']['f1']:>12.3f} "
                      f"{base_metrics['Comparator']['f1']:>12.3f} "
                      f"{'N/A':>10}")
            else:
                print(f"\n{'─'*80}")
                print(f"CASE: {case}")
                print(f"{'─'*80}")
                print("No precision data available")
        
        print("\n" + "="*80)
    
    
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
    
    def get_base_metrics(self, case: str, analysis_type: str) -> Dict:
        """
        Get all metrics (recall, precision, F1) for the base scenario.
        
        Args:
            case: Case name
            analysis_type: 'hpo' or 'con'
            
        Returns:
            Dictionary with metrics for each PICO element
        """
        key = f"{case}_{analysis_type}"
        
        if key not in self.results or not self.results[key]:
            return {}
        
        if 'base' not in self.results[key]:
            return {}
        
        return self.results[key]['base']