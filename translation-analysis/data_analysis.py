# -*- coding: utf-8 -*-
"""
Translation Data Analysis
Analyzes idiom translation quality across 4 systems (2 MT, 2 LLM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better output
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

# System categorization
MT_SYSTEMS = ['DeepL', 'Google Translate']
LLM_SYSTEMS = ['Gemini', 'ChatGPT']
ALL_SYSTEMS = MT_SYSTEMS + LLM_SYSTEMS

# Type labels
TYPE_LABELS = {
    1: '1-Literal',
    2: '2-Paraphrase', 
    3: '3-Cultural',
    4: '4-Other'
}

def load_and_parse_data(filepath):
    """Load ODS file and parse into structured format."""
    df = pd.read_excel(filepath, engine='odf')
    
    # Skip header row
    df = df.iloc[1:].reset_index(drop=True)
    
    # Find idiom rows and sentence/evaluation rows
    data_records = []
    
    current_idiom = None
    current_sentence_num = None
    
    for idx, row in df.iterrows():
        legenda = row['legenda']
        
        if legenda == 'idiom':
            current_idiom = row['Unnamed: 1']
        elif isinstance(legenda, str) and legenda.startswith('idiom_sentence'):
            current_sentence_num = int(legenda.replace('idiom_sentence', ''))
            # This row contains the translations
            for system in ALL_SYSTEMS:
                data_records.append({
                    'idiom': current_idiom,
                    'sentence_num': current_sentence_num,
                    'system': system,
                    'translation': row[system],
                    'original_sentence': row['Unnamed: 1']
                })
        elif pd.isna(legenda) and current_idiom is not None:
            # These are evaluation rows (yes/no or type 1-4)
            for system in ALL_SYSTEMS:
                val = row[system]
                # Find the matching record
                for rec in reversed(data_records):
                    if rec['idiom'] == current_idiom and rec['sentence_num'] == current_sentence_num and rec['system'] == system:
                        if str(val).lower() in ['yes', 'no', 'yes?', 'no?']:
                            rec['meaning'] = str(val).lower().replace('?', '')
                        elif str(val) in ['1', '2', '3', '4'] or val in [1, 2, 3, 4]:
                            rec['type'] = int(val)
                        break
    
    return pd.DataFrame(data_records)


def calculate_layer1_stats(df):
    """Calculate Layer 1 statistics: Meaning conveyance (yes/no)."""
    stats = {}
    
    # Overall per system
    meaning_counts = df.groupby('system')['meaning'].value_counts(normalize=True).unstack(fill_value=0)
    if 'yes' in meaning_counts.columns:
        stats['accuracy_per_system'] = (meaning_counts['yes'] * 100).to_dict()
    else:
        stats['accuracy_per_system'] = {sys: 0 for sys in ALL_SYSTEMS}
    
    # MT vs LLM
    df['system_type'] = df['system'].apply(lambda x: 'MT' if x in MT_SYSTEMS else 'LLM')
    type_counts = df.groupby('system_type')['meaning'].value_counts(normalize=True).unstack(fill_value=0)
    if 'yes' in type_counts.columns:
        stats['accuracy_mt_vs_llm'] = (type_counts['yes'] * 100).to_dict()
    else:
        stats['accuracy_mt_vs_llm'] = {'MT': 0, 'LLM': 0}
    
    # Per idiom per system
    idiom_stats = df.groupby(['idiom', 'system'])['meaning'].apply(
        lambda x: (x == 'yes').sum() / len(x) * 100
    ).unstack(fill_value=0)
    stats['per_idiom'] = idiom_stats
    
    # Raw counts
    stats['raw_counts'] = df.groupby('system')['meaning'].value_counts().unstack(fill_value=0)
    
    return stats

def calculate_layer2_stats(df):
    """Calculate Layer 2 statistics: Translation type distribution."""
    stats = {}
    
    # Filter rows with valid type
    df_with_type = df[df['type'].notna()]
    
    # Type distribution per system
    type_dist = df_with_type.groupby('system')['type'].value_counts(normalize=True).unstack(fill_value=0)
    stats['type_distribution'] = (type_dist * 100)
    
    # Type counts
    stats['type_counts'] = df_with_type.groupby('system')['type'].value_counts().unstack(fill_value=0)
    
    # MT vs LLM type preferences
    df_with_type['system_type'] = df_with_type['system'].apply(lambda x: 'MT' if x in MT_SYSTEMS else 'LLM')
    mt_llm_dist = df_with_type.groupby('system_type')['type'].value_counts(normalize=True).unstack(fill_value=0)
    stats['mt_vs_llm_types'] = (mt_llm_dist * 100)
    
    return stats

def calculate_combination_stats(df):
    """Calculate combination statistics: Meaning × Type."""
    stats = {}
    
    # Filter rows with both meaning and type
    df_complete = df[(df['meaning'].notna()) & (df['type'].notna())]
    
    # Cross-tabulation
    cross_tab = pd.crosstab(df_complete['meaning'], df_complete['type'], normalize='columns') * 100
    stats['meaning_by_type'] = cross_tab
    
    # Per system: success rate by type
    success_by_type = df_complete.groupby(['system', 'type'])['meaning'].apply(
        lambda x: (x == 'yes').sum() / len(x) * 100
    ).unstack(fill_value=0)
    stats['success_by_type_per_system'] = success_by_type
    
    return stats


def create_visualizations(df, layer1_stats, layer2_stats, combination_stats, output_dir):
    """Generate all visualizations."""
    
    # 1. Overall Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    systems = list(layer1_stats['accuracy_per_system'].keys())
    accuracies = [layer1_stats['accuracy_per_system'][s] for s in systems]
    colors = ['#2ecc71' if s in MT_SYSTEMS else '#3498db' for s in systems]
    
    bars = ax.bar(systems, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Layer 1: Meaning Conveyance Accuracy by System')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                '{:.1f}%'.format(acc), ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='MT Systems'),
                       Patch(facecolor='#3498db', label='LLM Systems')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. MT vs LLM Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    mt_llm_acc = layer1_stats['accuracy_mt_vs_llm']
    axes[0].bar(['MT Systems', 'LLM Systems'], [mt_llm_acc.get('MT', 0), mt_llm_acc.get('LLM', 0)],
                color=['#2ecc71', '#3498db'], edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Meaning Conveyance: MT vs LLM')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate([mt_llm_acc.get('MT', 0), mt_llm_acc.get('LLM', 0)]):
        axes[0].text(i, v + 1, '{:.1f}%'.format(v), ha='center', va='bottom', fontweight='bold')
    
    # Type distribution comparison
    mt_llm_types = layer2_stats['mt_vs_llm_types']
    x = np.arange(len(mt_llm_types.columns))
    width = 0.35
    
    if len(mt_llm_types) > 0:
        mt_vals = mt_llm_types.loc['MT'].values if 'MT' in mt_llm_types.index else [0]*4
        llm_vals = mt_llm_types.loc['LLM'].values if 'LLM' in mt_llm_types.index else [0]*4
        
        axes[1].bar(x - width/2, mt_vals, width, label='MT', color='#2ecc71', edgecolor='black')
        axes[1].bar(x + width/2, llm_vals, width, label='LLM', color='#3498db', edgecolor='black')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title('Translation Type Distribution: MT vs LLM')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([TYPE_LABELS.get(int(c), c) for c in mt_llm_types.columns])
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mt_vs_llm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Type Distribution per System (Stacked Bar)
    fig, ax = plt.subplots(figsize=(12, 7))
    type_dist = layer2_stats['type_distribution']
    
    if not type_dist.empty:
        type_dist_plot = type_dist.reindex(ALL_SYSTEMS)
        type_dist_plot.plot(kind='bar', stacked=True, ax=ax, 
                           colormap='viridis', edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Layer 2: Translation Type Distribution by System')
        ax.set_xlabel('System')
        ax.legend(title='Type', labels=[TYPE_LABELS.get(int(c), c) for c in type_dist.columns])
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'type_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Per-Idiom Heatmap
    fig, ax = plt.subplots(figsize=(14, 16))
    per_idiom = layer1_stats['per_idiom']
    
    if not per_idiom.empty:
        # Reorder columns
        per_idiom = per_idiom[ALL_SYSTEMS]
        sns.heatmap(per_idiom, annot=True, fmt='.0f', cmap='RdYlGn', 
                    center=50, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
        ax.set_title('Meaning Conveyance Accuracy by Idiom and System')
        ax.set_xlabel('System')
        ax.set_ylabel('Idiom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_idiom_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Combined Analysis: Success Rate by Type per System
    fig, ax = plt.subplots(figsize=(12, 7))
    success_by_type = combination_stats['success_by_type_per_system']
    
    if not success_by_type.empty:
        success_by_type = success_by_type.reindex(ALL_SYSTEMS)
        success_by_type.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Meaning Accuracy by Translation Type (Layer 1 × Layer 2)')
        ax.set_xlabel('System')
        ax.legend(title='Type', labels=[TYPE_LABELS.get(int(c), c) for c in success_by_type.columns])
        plt.xticks(rotation=0)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Individual System Detailed Breakdown
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, system in enumerate(ALL_SYSTEMS):
        ax = axes[idx]
        system_data = df[df['system'] == system]
        
        # Create pie chart for meaning
        meaning_counts = system_data['meaning'].value_counts()
        colors_pie = ['#2ecc71' if x == 'yes' else '#e74c3c' for x in meaning_counts.index]
        
        ax.pie(meaning_counts.values, labels=meaning_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax.set_title('{}\n({})'.format(system, 'MT' if system in MT_SYSTEMS else 'LLM'))
    
    plt.suptitle('Individual System Performance: Meaning Conveyance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_systems.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. Type Distribution Pie Charts per System
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, system in enumerate(ALL_SYSTEMS):
        ax = axes[idx]
        system_data = df[(df['system'] == system) & (df['type'].notna())]
        
        type_counts = system_data['type'].value_counts().sort_index()
        labels = [TYPE_LABELS.get(int(t), t) for t in type_counts.index]
        
        if len(type_counts) > 0:
            ax.pie(type_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('{}\n({})'.format(system, 'MT' if system in MT_SYSTEMS else 'LLM'))
    
    plt.suptitle('Translation Type Distribution by System', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'type_distribution_pies.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ All visualizations saved!")

def generate_report(df, layer1_stats, layer2_stats, combination_stats, output_dir):
    """Generate comprehensive markdown report."""
    
    report = []
    report.append("# Translation Data Analysis Report")
    report.append("\n*Generated on: {}*\n".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')))
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.append("- **Total idioms analyzed**: {}".format(df['idiom'].nunique()))
    report.append("- **Sentences per idiom**: {}".format(df.groupby('idiom')['sentence_num'].nunique().mode().values[0]))
    report.append("- **Total translations evaluated**: {}".format(len(df)))
    report.append("- **Systems analyzed**: {}".format(', '.join(ALL_SYSTEMS)))
    report.append("  - Machine Translation (MT): {}".format(', '.join(MT_SYSTEMS)))
    report.append("  - Large Language Models (LLM): {}\n".format(', '.join(LLM_SYSTEMS)))
    
    # Layer 1: Meaning Analysis
    report.append("---\n## Layer 1: Meaning Conveyance (Yes/No)\n")
    report.append("*Does the translation convey the meaning of the original idiom?*\n")
    
    report.append("### Overall Accuracy by System\n")
    report.append("| System | Category | Accuracy (%) | Yes Count | No Count |")
    report.append("|--------|----------|--------------|-----------|----------|")
    
    raw_counts = layer1_stats['raw_counts']
    for system in ALL_SYSTEMS:
        cat = 'MT' if system in MT_SYSTEMS else 'LLM'
        acc = layer1_stats['accuracy_per_system'].get(system, 0)
        yes_count = raw_counts.loc[system, 'yes'] if 'yes' in raw_counts.columns else 0
        no_count = raw_counts.loc[system, 'no'] if 'no' in raw_counts.columns else 0
        report.append("| {} | {} | {:.1f}% | {} | {} |".format(system, cat, acc, yes_count, no_count))
    
    report.append("\n### MT vs LLM Comparison\n")
    report.append("| Category | Accuracy (%) |")
    report.append("|----------|--------------|")
    for cat in ['MT', 'LLM']:
        acc = layer1_stats['accuracy_mt_vs_llm'].get(cat, 0)
        report.append("| {} | {:.1f}% |".format(cat, acc))
    
    # Ranking
    report.append("\n### System Ranking (by accuracy)\n")
    sorted_systems = sorted(layer1_stats['accuracy_per_system'].items(), key=lambda x: x[1], reverse=True)
    for rank, (system, acc) in enumerate(sorted_systems, 1):
        cat = 'MT' if system in MT_SYSTEMS else 'LLM'
        report.append("{}. **{}** ({}): {:.1f}%".format(rank, system, cat, acc))
    
    # Layer 2: Type Analysis
    report.append("\n---\n## Layer 2: Translation Type Distribution\n")
    report.append("*Translation strategy classification:*")
    report.append("- **Type 1**: Literal translation")
    report.append("- **Type 2**: Semantically accurate (paraphrasing)")
    report.append("- **Type 3**: Cultural equivalence")
    report.append("- **Type 4**: Other\n")
    
    report.append("### Type Distribution by System (%)\n")
    type_dist = layer2_stats['type_distribution']
    
    header = "| System |"
    for t in sorted(type_dist.columns):
        header += " Type {} |".format(int(t))
    report.append(header)
    
    sep = "|--------|"
    for _ in type_dist.columns:
        sep += "--------|"
    report.append(sep)
    
    for system in ALL_SYSTEMS:
        if system in type_dist.index:
            row = "| {} |".format(system)
            for t in sorted(type_dist.columns):
                val = type_dist.loc[system, t]
                row += " {:.1f}% |".format(val)
            report.append(row)
    
    report.append("\n### Type Counts by System\n")
    type_counts = layer2_stats['type_counts']
    
    header = "| System |"
    for t in sorted(type_counts.columns):
        header += " Type {} |".format(int(t))
    header += " Total |"
    report.append(header)
    
    sep = "|--------|"
    for _ in type_counts.columns:
        sep += "--------|"
    sep += "--------|"
    report.append(sep)
    
    for system in ALL_SYSTEMS:
        if system in type_counts.index:
            row = "| {} |".format(system)
            total = 0
            for t in sorted(type_counts.columns):
                val = int(type_counts.loc[system, t])
                row += " {} |".format(val)
                total += val
            row += " {} |".format(total)
            report.append(row)
    
    # Combination Analysis
    report.append("\n---\n## Combined Analysis: Meaning × Type\n")
    report.append("*Success rate of meaning conveyance by translation type*\n")
    
    success_by_type = combination_stats['success_by_type_per_system']
    
    if not success_by_type.empty:
        report.append("### Accuracy by Type per System (%)\n")
        header = "| System |"
        for t in sorted(success_by_type.columns):
            header += " Type {} |".format(int(t))
        report.append(header)
        
        sep = "|--------|"
        for _ in success_by_type.columns:
            sep += "--------|"
        report.append(sep)
        
        for system in ALL_SYSTEMS:
            if system in success_by_type.index:
                row = "| {} |".format(system)
                for t in sorted(success_by_type.columns):
                    val = success_by_type.loc[system, t]
                    row += " {:.1f}% |".format(val)
                report.append(row)
    
    
    # Individual System Analysis
    report.append("\n---\n## Individual System Analysis\n")
    
    for system in ALL_SYSTEMS:
        cat = 'MT' if system in MT_SYSTEMS else 'LLM'
        report.append("### {} ({})\n".format(system, cat))
        
        system_data = df[df['system'] == system]
        acc = layer1_stats['accuracy_per_system'].get(system, 0)
        
        report.append("- **Overall accuracy**: {:.1f}%".format(acc))
        
        # Best and worst idioms
        per_idiom = layer1_stats['per_idiom']
        if system in per_idiom.columns:
            best_idioms = per_idiom[system].nlargest(3)
            worst_idioms = per_idiom[system].nsmallest(3)
            
            report.append("- **Best performing idioms**:")
            for idiom, score in best_idioms.items():
                report.append("  - {}: {:.0f}%".format(idiom, score))
            
            report.append("- **Worst performing idioms**:")
            for idiom, score in worst_idioms.items():
                report.append("  - {}: {:.0f}%".format(idiom, score))
        
        # Type preferences
        if system in type_dist.index:
            dominant_type = type_dist.loc[system].idxmax()
            report.append("- **Dominant translation strategy**: Type {} ({}) - {:.1f}%".format(int(dominant_type), TYPE_LABELS.get(int(dominant_type), ''), type_dist.loc[system, dominant_type]))
        
        report.append("")
    
    # Key Findings
    report.append("---\n## Key Findings\n")
    
    # Best system
    best_system = max(layer1_stats['accuracy_per_system'].items(), key=lambda x: x[1])
    worst_system = min(layer1_stats['accuracy_per_system'].items(), key=lambda x: x[1])
    
    report.append("1. **Best performing system**: {} with {:.1f}% accuracy".format(best_system[0], best_system[1]))
    report.append("2. **Lowest performing system**: {} with {:.1f}% accuracy".format(worst_system[0], worst_system[1]))
    
    mt_avg = np.mean([layer1_stats['accuracy_per_system'][s] for s in MT_SYSTEMS if s in layer1_stats['accuracy_per_system']])
    llm_avg = np.mean([layer1_stats['accuracy_per_system'][s] for s in LLM_SYSTEMS if s in layer1_stats['accuracy_per_system']])
    
    if llm_avg > mt_avg:
        report.append("3. **LLMs outperform MT systems** on average ({:.1f}% vs {:.1f}%)".format(llm_avg, mt_avg))
    else:
        report.append("3. **MT systems outperform LLMs** on average ({:.1f}% vs {:.1f}%)".format(mt_avg, llm_avg))
    
    # Type patterns
    if not type_dist.empty:
        for system in ALL_SYSTEMS:
            if system in type_dist.index:
                type_2_pct = type_dist.loc[system, 2] if 2 in type_dist.columns else 0
                if type_2_pct > 50:
                    cat = 'MT' if system in MT_SYSTEMS else 'LLM'
                    report.append("4. **{}** ({}) primarily uses paraphrasing (Type 2: {:.1f}%)".format(system, cat, type_2_pct))
                    break
    
    # Visualizations reference
    report.append("\n---\n## Visualizations\n")
    report.append("The following visualization files have been generated:\n")
    report.append("- `overall_accuracy.png` - Bar chart of overall accuracy per system")
    report.append("- `mt_vs_llm_comparison.png` - Side-by-side MT vs LLM comparison")
    report.append("- `type_distribution.png` - Stacked bar chart of translation types")
    report.append("- `per_idiom_heatmap.png` - Heatmap of accuracy by idiom and system")
    report.append("- `combined_analysis.png` - Accuracy by translation type")
    report.append("- `individual_systems.png` - Pie charts for each system")
    report.append("- `type_distribution_pies.png` - Type distribution pie charts")
    
    # Write report
    report_path = output_dir / 'analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✓ Report saved to: {}".format(report_path))

def main():
    """Main analysis pipeline."""
    print("="*60)
    print("TRANSLATION DATA ANALYSIS")
    print("="*60)
    
    # Setup paths
    data_path = Path('Translations.xlsx.ods')
    output_dir = Path('.')
    
    # Load and parse data
    print("\n[1/6] Loading and parsing data...")
    df = load_and_parse_data(data_path)
    print("    Loaded {} translation records".format(len(df)))
    print("    Idioms: {}".format(df['idiom'].nunique()))
    print("    Systems: {}".format(df['system'].nunique()))
    
    
    # Calculate statistics
    print("\n[2/5] Calculating Layer 1 statistics (Meaning)...")
    layer1_stats = calculate_layer1_stats(df)
    
    print("\n[3/5] Calculating Layer 2 statistics (Type)...")
    layer2_stats = calculate_layer2_stats(df)
    
    print("\n[4/5] Calculating combination statistics...")
    combination_stats = calculate_combination_stats(df)
    
    # Generate outputs
    print("\n[5/5] Generating visualizations and report...")
    create_visualizations(df, layer1_stats, layer2_stats, combination_stats, output_dir)
    generate_report(df, layer1_stats, layer2_stats, combination_stats, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nOutput files generated:")
    print("  • analysis_report.md - Comprehensive statistical report")
    print("  • overall_accuracy.png")
    print("  • mt_vs_llm_comparison.png") 
    print("  • type_distribution.png")
    print("  • per_idiom_heatmap.png")
    print("  • combined_analysis.png")
    print("  • individual_systems.png")
    print("  • type_distribution_pies.png")

if __name__ == "__main__":
    main()
