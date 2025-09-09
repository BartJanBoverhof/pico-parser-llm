import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import os
from datetime import datetime
import matplotlib.patches as mpatches

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available. Install with: pip install geopandas")

class PICOAnalyzer:
    def __init__(self, pico_file_path):
        self.pico_file_path = pico_file_path
        self.data = None
        self.picos_df = None
        self.load_data()
        self.prepare_datamatrix()
    
    def load_data(self):
        try:
            with open(self.pico_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Successfully loaded PICO data from {self.pico_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"PICO file not found: {self.pico_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in PICO file: {e}")
        except Exception as e:
            raise Exception(f"Error loading PICO data: {e}")
    
    def prepare_datamatrix(self):
        pico_records = []
        
        for pico in self.data['consolidated_picos']:
            for country in pico['Countries']:
                for source_type in pico['Source_Types']:
                    record = {
                        'Population': pico['Population'],
                        'Intervention': pico['Intervention'],
                        'Comparator': pico['Comparator'],
                        'Country': country,
                        'Source_Type': source_type,
                        'Population_Variants_Count': len(pico['Original_Population_Variants']),
                        'Comparator_Variants_Count': len(pico['Original_Comparator_Variants'])
                    }
                    pico_records.append(record)
        
        self.picos_df = pd.DataFrame(pico_records)
    
    def print_summary_statistics(self):
        print("="*80)
        print("PICO ANALYSIS SUMMARY")
        print("="*80)
        
        metadata = self.data['consolidation_metadata']
        print(f"Analysis Date: {metadata['timestamp']}")
        print(f"Total Consolidated PICOs: {metadata['total_consolidated_picos']}")
        print(f"Source Countries: {', '.join(metadata['source_countries'])}")
        print(f"Source Types: {', '.join(metadata['source_types'])}")
        print()
        
        print("POPULATION AND COMPARATOR STATISTICS")
        print("-" * 50)
        
        country_counts = self.picos_df['Country'].value_counts()
        print("PICOs by Country:")
        for country, count in country_counts.items():
            print(f"  {country}: {count}")
        print()
        
        source_counts = self.picos_df['Source_Type'].value_counts()
        print("PICOs by Source Type:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        print()
        
        comparator_counts = self.picos_df['Comparator'].value_counts()
        print("Most Common Comparators:")
        for comp, count in comparator_counts.head(10).items():
            print(f"  {comp}: {count}")
        print()
        
        population_types = self.picos_df['Population'].nunique()
        print(f"Unique Population Types: {population_types}")
        print(f"Unique Comparators: {self.picos_df['Comparator'].nunique()}")
        print()
    
    def get_country_comparator_matrix(self):
        matrix = self.picos_df.pivot_table(
            index='Country', 
            columns='Comparator', 
            values='Intervention',
            aggfunc='count',
            fill_value=0
        )
        return matrix


class OutcomeAnalyzer:
    def __init__(self, outcome_file_path):
        self.outcome_file_path = outcome_file_path
        self.data = None
        self.outcomes_df = None
        self.load_data()
        self.prepare_datamatrix()
    
    def load_data(self):
        try:
            with open(self.outcome_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Successfully loaded Outcomes data from {self.outcome_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Outcomes file not found: {self.outcome_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in Outcomes file: {e}")
        except Exception as e:
            raise Exception(f"Error loading Outcomes data: {e}")
    
    def prepare_datamatrix(self):
        outcome_records = []
        
        for category, subcategories in self.data['consolidated_outcomes'].items():
            for subcategory, outcomes in subcategories.items():
                for outcome in outcomes:
                    if 'reported_by' in outcome:
                        for report in outcome['reported_by']:
                            record = {
                                'Category': category,
                                'Subcategory': subcategory,
                                'Outcome_Name': outcome['name'],
                                'Country': report['country'],
                                'Source_Type': report['source_type'],
                                'Has_Details': 'details' in outcome and bool(outcome.get('details', []))
                            }
                            outcome_records.append(record)
                    else:
                        record = {
                            'Category': category,
                            'Subcategory': subcategory,
                            'Outcome_Name': outcome['name'],
                            'Country': 'Unknown',
                            'Source_Type': 'Unknown',
                            'Has_Details': 'details' in outcome and bool(outcome.get('details', []))
                        }
                        outcome_records.append(record)
        
        self.outcomes_df = pd.DataFrame(outcome_records)
    
    def print_summary_statistics(self):
        print("="*80)
        print("OUTCOMES ANALYSIS SUMMARY")
        print("="*80)
        
        metadata = self.data['outcomes_metadata']
        print(f"Analysis Date: {metadata['timestamp']}")
        print(f"Total Unique Outcomes: {metadata['total_unique_outcomes']}")
        print(f"Source Countries: {', '.join(metadata['source_countries'])}")
        print(f"Source Types: {', '.join(metadata['source_types'])}")
        print()
        
        print("OUTCOMES STATISTICS")
        print("-" * 50)
        
        category_counts = self.outcomes_df['Category'].value_counts()
        print("Outcomes by Category:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        print()
        
        country_outcome_counts = self.outcomes_df['Country'].value_counts()
        print("Outcome Reports by Country:")
        for country, count in country_outcome_counts.items():
            print(f"  {country}: {count}")
        print()
        
        source_outcome_counts = self.outcomes_df['Source_Type'].value_counts()
        print("Outcome Reports by Source Type:")
        for source, count in source_outcome_counts.items():
            print(f"  {source}: {count}")
        print()
        
        print("Most Frequently Reported Outcomes:")
        outcome_frequency = self.outcomes_df['Outcome_Name'].value_counts()
        for outcome, count in outcome_frequency.head(10).items():
            print(f"  {outcome}: {count}")
        print()
    
    def get_category_country_matrix(self):
        matrix = self.outcomes_df.pivot_table(
            index='Category', 
            columns='Country', 
            values='Outcome_Name',
            aggfunc='count',
            fill_value=0
        )
        return matrix
    
    def get_outcome_source_matrix(self):
        matrix = self.outcomes_df.pivot_table(
            index='Outcome_Name', 
            columns='Source_Type', 
            values='Country',
            aggfunc='count',
            fill_value=0
        )
        return matrix


class DataVisualizer:
    def __init__(self, pico_analyzer, outcome_analyzer, output_dir="results/visualizations"):
        self.pico_analyzer = pico_analyzer
        self.outcome_analyzer = outcome_analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.2
        })
        
        sns.set_style("whitegrid", {
            'axes.grid': True,
            'grid.color': '.8',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5
        })
        
        self.scientific_colors = {
            'primary': '#2E4057',
            'secondary': '#048A81',
            'tertiary': '#F18F01',
            'quaternary': '#C73E1D',
            'light_gray': '#E8E8E8',
            'dark_gray': '#4A4A4A'
        }
    
    def create_european_heatmap(self):
        print("Creating European PICO distribution map...")
        
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Creating simplified map...")
            self._create_simplified_european_map()
            return
            
        try:
            print("Loading geographic data...")
            
            world_url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/world-110m2.json"
            
            try:
                world = gpd.read_file(world_url)
                print("Successfully loaded world data from web")
            except:
                print("Could not load from web, trying alternative dataset...")
                world_url_alt = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
                try:
                    world = gpd.read_file(world_url_alt)
                    print("Successfully loaded alternative dataset")
                except:
                    print("Could not load geographic data, falling back to simplified map")
                    self._create_simplified_european_map()
                    return
            
            country_mapping = {
                'DE': 'Germany',
                'DK': 'Denmark', 
                'EN': 'United Kingdom',
                'FR': 'France',
                'NL': 'Netherlands',
                'PO': 'Poland',
                'PT': 'Portugal',
                'SE': 'Sweden'
            }
            
            name_column = 'NAME' if 'NAME' in world.columns else 'name'
            if name_column not in world.columns:
                for col in ['NAME_EN', 'NAME_LONG', 'ADMIN', 'Country']:
                    if col in world.columns:
                        name_column = col
                        break
            
            country_counts = self.pico_analyzer.picos_df['Country'].value_counts()
            print(f"Countries with PICO data: {dict(country_counts)}")
            
            pico_data = []
            for country_code, count in country_counts.items():
                if country_code in country_mapping:
                    country_name = country_mapping[country_code]
                    pico_data.append({name_column: country_name, 'pico_count': count, 'code': country_code})
            
            pico_df = pd.DataFrame(pico_data)
            
            european_countries = [
                'Germany', 'Denmark', 'United Kingdom', 'France', 'Netherlands', 
                'Poland', 'Portugal', 'Sweden', 'Norway', 'Finland', 'Spain', 
                'Italy', 'Switzerland', 'Austria', 'Belgium', 'Ireland',
                'Czech Republic', 'Hungary', 'Romania', 'Bulgaria', 'Greece',
                'Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'Latvia', 'Estonia'
            ]
            
            europe = world[world[name_column].isin(european_countries)].copy()
            
            europe = europe.merge(pico_df, on=name_column, how='left')
            europe['pico_count'] = europe['pico_count'].fillna(0)
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            ax.set_xlim(-12, 32)
            ax.set_ylim(35, 72)
            
            europe_no_data = europe[europe['pico_count'] == 0]
            if not europe_no_data.empty:
                europe_no_data.plot(ax=ax, color='#F0F0F0', edgecolor='white', linewidth=0.8)
            
            europe_with_data = europe[europe['pico_count'] > 0]
            if not europe_with_data.empty:
                vmin = europe_with_data['pico_count'].min()
                vmax = europe_with_data['pico_count'].max()
                
                europe_with_data.plot(
                    column='pico_count', 
                    ax=ax, 
                    cmap='Blues',
                    legend=False,
                    edgecolor='white', 
                    linewidth=0.8,
                    vmin=vmin,
                    vmax=vmax
                )
                
                for idx, row in europe_with_data.iterrows():
                    centroid = row.geometry.centroid
                    ax.annotate(
                        f"{row['code']}\n({int(row['pico_count'])})",
                        xy=(centroid.x, centroid.y),
                        ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.8, edgecolor='none'),
                        zorder=5
                    )
                
                sm = plt.cm.ScalarMappable(cmap='Blues', 
                                         norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30)
                cbar.set_label('Number of PICOs', fontweight='bold', fontsize=12)
                cbar.ax.tick_params(labelsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            ax.set_title('European Distribution of PICO Evidence\nby Country', 
                        fontsize=16, fontweight='bold', pad=20)
            
            no_data_patch = mpatches.Patch(color='#F0F0F0', label='No PICO data')
            ax.legend(handles=[no_data_patch], loc='upper left', 
                     bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'european_pico_heatmap.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            print("European choropleth map saved successfully")
            plt.show()
            
        except Exception as e:
            print(f"Error creating European map with geopandas: {str(e)}")
            print("Falling back to simplified map...")
            self._create_simplified_european_map()
    
    def _create_simplified_european_map(self):
        print("Creating simplified European map...")
        
        country_counts = self.pico_analyzer.picos_df['Country'].value_counts()
        
        country_mapping = {
            'DE': 'Germany',
            'DK': 'Denmark', 
            'EN': 'United Kingdom',
            'FR': 'France',
            'NL': 'Netherlands',
            'PO': 'Poland',
            'PT': 'Portugal',
            'SE': 'Sweden'
        }
        
        european_positions = {
            'Germany': (10.5, 51.5),
            'Denmark': (10.0, 56.0),
            'United Kingdom': (-2.0, 54.0),
            'France': (2.5, 46.5),
            'Netherlands': (5.5, 52.0),
            'Poland': (19.0, 52.0),
            'Portugal': (-8.0, 39.5),
            'Sweden': (15.0, 62.0)
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-12, 25)
        ax.set_ylim(35, 70)
        
        ax.add_patch(plt.Rectangle((-10, 35), 35, 35, 
                                 facecolor='#F8F9FA', edgecolor='#E9ECEF', 
                                 alpha=0.5, linewidth=1))
        
        if len(country_counts) == 0:
            print("Warning: No country data found for heatmap")
            return
            
        max_count = country_counts.max()
        min_count = country_counts.min()
        
        for country_code, count in country_counts.items():
            if country_code in country_mapping:
                country_name = country_mapping[country_code]
                if country_name in european_positions:
                    lon, lat = european_positions[country_name]
                    
                    if max_count > min_count:
                        normalized_size = (count - min_count) / (max_count - min_count)
                    else:
                        normalized_size = 1.0
                        
                    circle_size = 300 + normalized_size * 700
                    color_intensity = 0.3 + 0.7 * normalized_size
                    
                    ax.scatter(lon, lat, s=circle_size, 
                             color=plt.cm.Blues(color_intensity), 
                             alpha=0.8, edgecolors='white', linewidth=2,
                             zorder=5)
                    
                    ax.annotate(f'{country_code}\n({count})', 
                               (lon, lat), xytext=(0, 0), 
                               textcoords='offset points',
                               ha='center', va='center',
                               fontsize=10, fontweight='bold',
                               color='white' if normalized_size > 0.5 else 'black',
                               zorder=6)
        
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        ax.set_title('European Distribution of PICO Evidence\n(Simplified View)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'european_pico_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Simplified European map saved successfully")
        plt.show()
    
    def create_pico_visualizations(self):
        print("Creating PICO visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PICO Analysis Overview', fontsize=16, fontweight='bold', y=0.95)
        
        country_counts = self.pico_analyzer.picos_df['Country'].value_counts()
        bars1 = axes[0, 0].bar(range(len(country_counts)), country_counts.values, 
                              color=self.scientific_colors['primary'], alpha=0.8,
                              edgecolor='white', linewidth=0.8)
        axes[0, 0].set_xticks(range(len(country_counts)))
        axes[0, 0].set_xticklabels(country_counts.index, fontweight='bold')
        axes[0, 0].set_ylabel('Number of PICOs', fontweight='bold')
        axes[0, 0].set_xlabel('Country', fontweight='bold')
        axes[0, 0].set_title('A. PICO Distribution by Country', fontweight='bold', pad=15)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars1, country_counts.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        source_counts = self.pico_analyzer.picos_df['Source_Type'].value_counts()
        colors = [self.scientific_colors['secondary'], self.scientific_colors['tertiary']]
        wedges, texts, autotexts = axes[0, 1].pie(source_counts.values, 
                                                 labels=source_counts.index,
                                                 autopct='%1.1f%%', 
                                                 colors=colors,
                                                 startangle=90,
                                                 wedgeprops=dict(edgecolor='white', linewidth=2))
        axes[0, 1].set_title('B. PICO Distribution by Source Type', fontweight='bold', pad=15)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        comp_counts = self.pico_analyzer.picos_df['Comparator'].value_counts().head(8)
        bars2 = axes[1, 0].barh(range(len(comp_counts)), comp_counts.values, 
                               color=self.scientific_colors['quaternary'], alpha=0.8,
                               edgecolor='white', linewidth=0.8)
        axes[1, 0].set_yticks(range(len(comp_counts)))
        axes[1, 0].set_yticklabels([comp[:25] + '...' if len(comp) > 25 else comp 
                                   for comp in comp_counts.index], fontweight='bold')
        axes[1, 0].set_xlabel('Number of PICOs', fontweight='bold')
        axes[1, 0].set_title('C. Most Frequent Comparators', fontweight='bold', pad=15)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        axes[1, 0].invert_yaxis()
        
        for bar, value in zip(bars2, comp_counts.values):
            axes[1, 0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                           str(value), ha='left', va='center', fontweight='bold')
        
        country_source_matrix = self.pico_analyzer.picos_df.pivot_table(
            index='Country', columns='Source_Type', values='Intervention', 
            aggfunc='count', fill_value=0
        )
        
        im = axes[1, 1].imshow(country_source_matrix.values, cmap='Blues', aspect='auto',
                               vmin=0, vmax=country_source_matrix.values.max())
        
        axes[1, 1].set_xticks(range(len(country_source_matrix.columns)))
        axes[1, 1].set_xticklabels([col.replace('_', '\n') for col in country_source_matrix.columns], 
                                  fontweight='bold')
        axes[1, 1].set_yticks(range(len(country_source_matrix.index)))
        axes[1, 1].set_yticklabels(country_source_matrix.index, fontweight='bold')
        axes[1, 1].set_title('D. Country vs Source Type Matrix', fontweight='bold', pad=15)
        
        for i in range(len(country_source_matrix.index)):
            for j in range(len(country_source_matrix.columns)):
                value = country_source_matrix.iloc[i, j]
                axes[1, 1].text(j, i, str(value), ha='center', va='center',
                               fontweight='bold', 
                               color='white' if value > country_source_matrix.values.max()/2 else 'black')
        
        cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        cbar.set_label('Number of PICOs', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pico_analysis_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("Creating additional PICO visualizations...")
        self._create_comparator_heatmap()
        self.create_european_heatmap()
    
    def _create_comparator_heatmap(self):
        matrix = self.pico_analyzer.get_country_comparator_matrix()
        
        plt.figure(figsize=(12, 8))
        
        mask = matrix == 0
        ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                        cbar_kws={'label': 'Number of PICOs'},
                        linewidths=0.5, linecolor='white',
                        mask=None, square=False)
        
        plt.title('Country vs Comparator Distribution Matrix', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Comparator', fontweight='bold')
        plt.ylabel('Country', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.yticks(rotation=0, fontweight='bold')
        
        cbar = ax.collections[0].colorbar
        cbar.set_label('Number of PICOs', fontweight='bold')
        
        for text in ax.texts:
            text.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'country_comparator_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_outcome_visualizations(self):
        print("Creating Outcome visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Outcomes Analysis Overview', fontsize=16, fontweight='bold', y=0.95)
        
        category_counts = self.outcome_analyzer.outcomes_df['Category'].value_counts()
        bars1 = axes[0, 0].bar(range(len(category_counts)), category_counts.values, 
                              color=self.scientific_colors['primary'], alpha=0.8,
                              edgecolor='white', linewidth=0.8)
        axes[0, 0].set_xticks(range(len(category_counts)))
        axes[0, 0].set_xticklabels([cat.replace('_', ' ').title() for cat in category_counts.index], 
                                  rotation=45, ha='right', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Outcome Reports', fontweight='bold')
        axes[0, 0].set_xlabel('Outcome Category', fontweight='bold')
        axes[0, 0].set_title('A. Outcomes by Category', fontweight='bold', pad=15)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars1, category_counts.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        country_counts = self.outcome_analyzer.outcomes_df['Country'].value_counts()
        bars2 = axes[0, 1].bar(range(len(country_counts)), country_counts.values, 
                              color=self.scientific_colors['secondary'], alpha=0.8,
                              edgecolor='white', linewidth=0.8)
        axes[0, 1].set_xticks(range(len(country_counts)))
        axes[0, 1].set_xticklabels(country_counts.index, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Outcome Reports', fontweight='bold')
        axes[0, 1].set_xlabel('Country', fontweight='bold')
        axes[0, 1].set_title('B. Outcome Reports by Country', fontweight='bold', pad=15)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, country_counts.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        source_counts = self.outcome_analyzer.outcomes_df['Source_Type'].value_counts()
        colors = [self.scientific_colors['tertiary'], self.scientific_colors['quaternary']]
        wedges, texts, autotexts = axes[1, 0].pie(source_counts.values, 
                                                 labels=source_counts.index,
                                                 autopct='%1.1f%%', 
                                                 colors=colors,
                                                 startangle=90,
                                                 wedgeprops=dict(edgecolor='white', linewidth=2))
        axes[1, 0].set_title('C. Outcome Reports by Source Type', fontweight='bold', pad=15)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        category_country_matrix = self.outcome_analyzer.get_category_country_matrix()
        
        im = axes[1, 1].imshow(category_country_matrix.values, cmap='Greens', aspect='auto',
                               vmin=0, vmax=category_country_matrix.values.max())
        
        axes[1, 1].set_xticks(range(len(category_country_matrix.columns)))
        axes[1, 1].set_xticklabels(category_country_matrix.columns, fontweight='bold', rotation=45, ha='right')
        axes[1, 1].set_yticks(range(len(category_country_matrix.index)))
        axes[1, 1].set_yticklabels([idx.replace('_', ' ').title() for idx in category_country_matrix.index], 
                                  fontweight='bold')
        axes[1, 1].set_title('D. Category vs Country Matrix', fontweight='bold', pad=15)
        
        for i in range(len(category_country_matrix.index)):
            for j in range(len(category_country_matrix.columns)):
                value = category_country_matrix.iloc[i, j]
                axes[1, 1].text(j, i, str(value), ha='center', va='center',
                               fontweight='bold',
                               color='white' if value > category_country_matrix.values.max()/2 else 'black')
        
        cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        cbar.set_label('Number of Reports', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outcomes_analysis_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        self._create_outcome_frequency_plot()
    
    def _create_outcome_frequency_plot(self):
        outcome_freq = self.outcome_analyzer.outcomes_df['Outcome_Name'].value_counts().head(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(outcome_freq)), outcome_freq.values, 
                       color=self.scientific_colors['primary'], alpha=0.8,
                       edgecolor='white', linewidth=0.8)
        
        plt.yticks(range(len(outcome_freq)), 
                  [outcome[:40] + '...' if len(outcome) > 40 else outcome 
                   for outcome in outcome_freq.index], fontweight='bold')
        plt.xlabel('Number of Reports', fontweight='bold')
        plt.ylabel('Outcome Measure', fontweight='bold')
        plt.title('Most Frequently Reported Outcome Measures', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, value) in enumerate(zip(bars, outcome_freq.values)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(value)}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outcome_frequency_plot.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_combined_analysis(self):
        print("Creating combined analysis visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Source Type Distribution Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        pico_sources = self.pico_analyzer.picos_df['Source_Type'].value_counts()
        colors_pico = [self.scientific_colors['primary'], self.scientific_colors['secondary']]
        wedges1, texts1, autotexts1 = axes[0].pie(pico_sources.values, 
                                                  labels=[label.replace('_', ' ').title() 
                                                         for label in pico_sources.index],
                                                  autopct='%1.1f%%', 
                                                  colors=colors_pico,
                                                  startangle=90,
                                                  wedgeprops=dict(edgecolor='white', linewidth=2))
        axes[0].set_title('A. PICOs by Source Type', fontweight='bold', pad=20)
        
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        outcome_sources = self.outcome_analyzer.outcomes_df['Source_Type'].value_counts()
        colors_outcome = [self.scientific_colors['tertiary'], self.scientific_colors['quaternary']]
        wedges2, texts2, autotexts2 = axes[1].pie(outcome_sources.values, 
                                                  labels=[label.replace('_', ' ').title() 
                                                         for label in outcome_sources.index],
                                                  autopct='%1.1f%%', 
                                                  colors=colors_outcome,
                                                  startangle=90,
                                                  wedgeprops=dict(edgecolor='white', linewidth=2))
        axes[1].set_title('B. Outcomes by Source Type', fontweight='bold', pad=20)
        
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'source_type_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        self._create_country_coverage_comparison()
    
    def _create_country_coverage_comparison(self):
        pico_countries = set(self.pico_analyzer.picos_df['Country'].unique())
        outcome_countries = set(self.outcome_analyzer.outcomes_df['Country'].unique())
        
        all_countries = sorted(pico_countries.union(outcome_countries))
        
        pico_coverage = [1 if country in pico_countries else 0 for country in all_countries]
        outcome_coverage = [1 if country in outcome_countries else 0 for country in all_countries]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(all_countries))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pico_coverage, width, label='PICOs', 
                      color=self.scientific_colors['primary'], alpha=0.8,
                      edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + width/2, outcome_coverage, width, label='Outcomes', 
                      color=self.scientific_colors['secondary'], alpha=0.8,
                      edgecolor='white', linewidth=0.8)
        
        ax.set_xlabel('Country', fontweight='bold')
        ax.set_ylabel('Data Availability (1 = Available, 0 = Not Available)', fontweight='bold')
        ax.set_title('Country Data Coverage: PICOs vs Outcomes', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(all_countries, fontweight='bold')
        legend = ax.legend(frameon=True, fancybox=True, shadow=True)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'country_coverage_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def generate_summary_report(self):
        print("Generating summary report...")
        
        report_content = []
        report_content.append("RAG PIPELINE ANALYSIS SUMMARY REPORT")
        report_content.append("=" * 50)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        report_content.append("PICO ANALYSIS SUMMARY:")
        report_content.append("-" * 25)
        report_content.append(f"Total consolidated PICOs: {len(self.pico_analyzer.data['consolidated_picos'])}")
        report_content.append(f"Unique countries: {self.pico_analyzer.picos_df['Country'].nunique()}")
        report_content.append(f"Unique comparators: {self.pico_analyzer.picos_df['Comparator'].nunique()}")
        report_content.append(f"Most common country: {self.pico_analyzer.picos_df['Country'].mode()[0]}")
        report_content.append(f"Most common comparator: {self.pico_analyzer.picos_df['Comparator'].mode()[0]}")
        report_content.append("")
        
        report_content.append("OUTCOMES ANALYSIS SUMMARY:")
        report_content.append("-" * 27)
        report_content.append(f"Total outcome reports: {len(self.outcome_analyzer.outcomes_df)}")
        report_content.append(f"Unique outcomes: {self.outcome_analyzer.outcomes_df['Outcome_Name'].nunique()}")
        report_content.append(f"Outcome categories: {self.outcome_analyzer.outcomes_df['Category'].nunique()}")
        report_content.append(f"Most reported outcome: {self.outcome_analyzer.outcomes_df['Outcome_Name'].mode()[0]}")
        report_content.append(f"Most active country: {self.outcome_analyzer.outcomes_df['Country'].mode()[0]}")
        report_content.append("")
        
        pico_countries = set(self.pico_analyzer.picos_df['Country'].unique())
        outcome_countries = set(self.outcome_analyzer.outcomes_df['Country'].unique())
        common_countries = pico_countries.intersection(outcome_countries)
        
        report_content.append("COVERAGE ANALYSIS:")
        report_content.append("-" * 18)
        report_content.append(f"Countries with both PICOs and outcomes: {len(common_countries)}")
        report_content.append(f"PICO-only countries: {len(pico_countries - outcome_countries)}")
        report_content.append(f"Outcome-only countries: {len(outcome_countries - pico_countries)}")
        report_content.append(f"Total country coverage: {len(pico_countries.union(outcome_countries))}")
        
        with open(self.output_dir / 'analysis_summary_report.txt', 'w') as f:
            f.write('\n'.join(report_content))
        
        print('\n'.join(report_content))


def run_complete_analysis(pico_file_path, outcome_file_path):
    print("Starting comprehensive RAG pipeline analysis...")
    print()
    
    pico_analyzer = PICOAnalyzer(pico_file_path)
    outcome_analyzer = OutcomeAnalyzer(outcome_file_path)
    
    pico_analyzer.print_summary_statistics()
    outcome_analyzer.print_summary_statistics()
    
    visualizer = DataVisualizer(pico_analyzer, outcome_analyzer)
    
    visualizer.create_pico_visualizations()
    visualizer.create_outcome_visualizations()
    visualizer.create_combined_analysis()
    
    visualizer.generate_summary_report()
    
    print("Analysis complete! All visualizations and reports saved to results/visualizations/")
    
    return pico_analyzer, outcome_analyzer, visualizer