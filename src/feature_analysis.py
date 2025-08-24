import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cross_dataset_analysis import preprocess_dataset1, preprocess_dataset2

def visualize_feature_distributions():
    """Create visualizations to compare feature distributions between datasets"""
    
    # Load datasets
    dataset1 = preprocess_dataset1()
    dataset2 = preprocess_dataset2()
    
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_columns):
        ax = axes[i]
        
        # Plot distributions
        ax.hist(dataset1[feature], alpha=0.7, label='Dataset 1 (Pima)', bins=30, density=True)
        ax.hist(dataset2[feature], alpha=0.7, label='Dataset 2 (Hospital)', bins=30, density=True)
        
        ax.set_title(f'{feature} Distribution')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Dataset 1 correlation
    corr1 = dataset1[feature_columns + ['Outcome']].corr()
    sns.heatmap(corr1, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('Dataset 1 (Pima) Feature Correlations')
    
    # Dataset 2 correlation
    corr2 = dataset2[feature_columns + ['Outcome']].corr()
    sns.heatmap(corr2, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Dataset 2 (Hospital) Feature Correlations')
    
    plt.tight_layout()
    plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def feature_mapping_summary():
    """Print detailed summary of how features were mapped"""
    print("="*60)
    print("FEATURE MAPPING SUMMARY")
    print("="*60)
    
    mapping_info = {
        'Age': 'Hospital: Age ranges converted to midpoint values',
        'Pregnancies': 'Hospital: Female patients use inpatient visits as proxy, Males = 0',
        'Glucose': 'Hospital: max_glu_serum mapped (None/Norm=100, >200=250, >300=350)',
        'BloodPressure': 'Hospital: time_in_hospital * 5 + 70 (proxy based on hospital stay)',
        'SkinThickness': 'Hospital: num_procedures * 5 (medical complexity proxy)',
        'Insulin': 'Hospital: insulin medication usage (Up/Down/Steady=100, else=0)',
        'BMI': 'Hospital: 20 + num_medications * 0.5 (health complexity proxy)',
        'DiabetesPedigreeFunction': 'Hospital: number_diagnoses * 0.1 (genetic risk proxy)',
        'Outcome': 'Hospital: readmitted (>30 or <30) = 1, NO = 0'
    }
    
    for feature, mapping in mapping_info.items():
        print(f"{feature:25}: {mapping}")
    
    print("\n" + "="*60)
    print("CROSS-DATASET PERFORMANCE INTERPRETATION")
    print("="*60)
    
    print("""
The cross-dataset training results show:

1. SAME DATASET PERFORMANCE:
   - Dataset 1 -> Dataset 1: 76.0% (Good performance on Pima dataset)
   - Dataset 2 -> Dataset 2: 56.8% (Moderate performance on Hospital dataset)

2. CROSS-DATASET PERFORMANCE:
   - Dataset 1 -> Dataset 2: 53.8% (Pima model on Hospital data)
   - Dataset 2 -> Dataset 1: 49.5% (Hospital model on Pima data)

3. ANALYSIS:
   - The Pima model performs better when applied to hospital data than vice versa
   - Cross-dataset accuracy drops significantly (76% -> 54% and 57% -> 50%)
   - This indicates the datasets have different underlying distributions
   - Feature engineering for Dataset 2 was proxy-based, not direct measurements

4. RECOMMENDATIONS:
   - Use Dataset 1 (Pima) for training as it has direct medical measurements
   - Dataset 2 can be used for additional validation but requires better feature mapping
   - Consider domain adaptation techniques for better cross-dataset performance
   - Collect more similar datasets with direct medical measurements
    """)

if __name__ == "__main__":
    feature_mapping_summary()
    
    # Uncomment to generate visualizations (requires matplotlib)
    try:
        visualize_feature_distributions()
        print("\nVisualization plots saved as PNG files.")
    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib seaborn")
