#!/usr/bin/env python3
"""
MedBot System Evaluation Against FAQ Test Cases
Generates visual results and metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_faq_data():
    """Load FAQ test data"""
    try:
        df = pd.read_csv('FAQ_Test.csv')
        print(f"✅ Loaded {len(df)} test questions from FAQ_Test.csv")
        return df
    except Exception as e:
        print(f"❌ Error loading FAQ_Test.csv: {e}")
        return None

def test_rag_system():
    """Test RAG system by calling the Flask app"""
    print("\n🔄 Testing RAG System...")
    
    # Load FAQ data
    df = load_faq_data()
    if df is None:
        return None
    
    results = []
    
    # Test first 10 questions (to avoid rate limits)
    test_questions = df.head(10)
    
    for idx, row in test_questions.iterrows():
        question = row['Question']
        expected = row['Expected_answer']
        
        print(f"Testing Q{idx+1}: {question[:50]}...")
        
        try:
            # Call the Flask app
            response = requests.post('http://localhost:5000/ask', 
                                   json={'question': question}, 
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract answers
                rag_answer = data.get('rag', 'No answer')
                biogpt_answer = data.get('biogpt', 'No answer')
                clinbert_answer = data.get('clinbert', 'No answer')
                github_answer = data.get('gemini_backup', 'No answer')
                
                results.append({
                    'question': question,
                    'expected': expected,
                    'rag_answer': rag_answer,
                    'biogpt_answer': biogpt_answer,
                    'clinbert_answer': clinbert_answer,
                    'github_answer': github_answer,
                    'status': 'success'
                })
                
            else:
                print(f"❌ API error: {response.status_code}")
                results.append({
                    'question': question,
                    'expected': expected,
                    'rag_answer': 'API Error',
                    'biogpt_answer': 'API Error',
                    'clinbert_answer': 'API Error',
                    'github_answer': 'API Error',
                    'status': 'error'
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'question': question,
                'expected': expected,
                'rag_answer': 'Connection Error',
                'biogpt_answer': 'Connection Error',
                'clinbert_answer': 'Connection Error',
                'github_answer': 'Connection Error',
                'status': 'error'
            })
    
    return pd.DataFrame(results)

def calculate_semantic_similarity(text1, text2, model):
    """Calculate semantic similarity between two texts"""
    try:
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity * 100  # Convert to percentage
    except:
        return 0

def evaluate_answers(results_df):
    """Evaluate answers using semantic similarity"""
    print("\n📊 Calculating semantic similarities...")
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate similarities
    rag_similarities = []
    biogpt_similarities = []
    clinbert_similarities = []
    github_similarities = []
    
    for idx, row in results_df.iterrows():
        if row['status'] == 'success':
            expected = row['expected']
            
            rag_sim = calculate_semantic_similarity(expected, row['rag_answer'], model)
            biogpt_sim = calculate_semantic_similarity(expected, row['biogpt_answer'], model)
            clinbert_sim = calculate_semantic_similarity(expected, row['clinbert_answer'], model)
            github_sim = calculate_semantic_similarity(expected, row['github_answer'], model)
            
            rag_similarities.append(rag_sim)
            biogpt_similarities.append(biogpt_sim)
            clinbert_similarities.append(clinbert_sim)
            github_similarities.append(github_sim)
            
            print(f"Q{idx+1}: RAG={rag_sim:.1f}%, BioGPT={biogpt_sim:.1f}%, Clinical-BERT={clinbert_sim:.1f}%, GitHub={github_sim:.1f}%")
        else:
            rag_similarities.append(0)
            biogpt_similarities.append(0)
            clinbert_similarities.append(0)
            github_similarities.append(0)
    
    return {
        'rag': rag_similarities,
        'biogpt': biogpt_similarities,
        'clinbert': clinbert_similarities,
        'github': github_similarities
    }

def create_evaluation_plots(similarities):
    """Create evaluation plots"""
    print("\n📈 Creating evaluation plots...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MedBot System Evaluation Results\nSemantic Similarity vs FAQ Expected Answers', 
                 fontsize=16, fontweight='bold')
    
    # Data for plotting
    components = ['RAG System', 'BioGPT', 'Clinical-BERT', 'GitHub AI']
    avg_scores = [
        np.mean(similarities['rag']),
        np.mean(similarities['biogpt']),
        np.mean(similarities['clinbert']),
        np.mean(similarities['github'])
    ]
    
    colors = ['#9b59b6', '#4ecdc4', '#45b7d1', '#667eea']
    
    # 1. Average Scores Bar Chart
    bars = ax1.bar(components, avg_scores, color=colors, alpha=0.8)
    ax1.set_title('Average Semantic Similarity Scores', fontweight='bold')
    ax1.set_ylabel('Similarity Score (%)')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Box Plot showing distribution
    data_for_box = [similarities['rag'], similarities['biogpt'], 
                    similarities['clinbert'], similarities['github']]
    
    box_plot = ax2.boxplot(data_for_box, labels=components, patch_artist=True)
    ax2.set_title('Score Distribution (Box Plot)', fontweight='bold')
    ax2.set_ylabel('Similarity Score (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 3. Question-by-Question Comparison
    questions = range(1, len(similarities['rag']) + 1)
    ax3.plot(questions, similarities['rag'], 'o-', label='RAG System', color=colors[0], linewidth=2)
    ax3.plot(questions, similarities['biogpt'], 's-', label='BioGPT', color=colors[1], linewidth=2)
    ax3.plot(questions, similarities['clinbert'], '^-', label='Clinical-BERT', color=colors[2], linewidth=2)
    ax3.plot(questions, similarities['github'], 'd-', label='GitHub AI', color=colors[3], linewidth=2)
    
    ax3.set_title('Question-by-Question Performance', fontweight='bold')
    ax3.set_xlabel('Question Number')
    ax3.set_ylabel('Similarity Score (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Summary Table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for i, component in enumerate(components):
        scores = [similarities['rag'], similarities['biogpt'], 
                 similarities['clinbert'], similarities['github']][i]
        summary_data.append([
            component,
            f"{np.mean(scores):.1f}%",
            f"{np.std(scores):.1f}%",
            f"{np.min(scores):.1f}%",
            f"{np.max(scores):.1f}%"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Component', 'Mean', 'Std Dev', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(components)):
        table[(i+1, 0)].set_facecolor(colors[i])
        table[(i+1, 0)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'MedBot_Evaluation_Results_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Evaluation plot saved as: {filename}")
    
    plt.show()
    
    return filename

def generate_detailed_report(results_df, similarities):
    """Generate detailed evaluation report"""
    print("\n📝 Generating detailed report...")
    
    report = []
    report.append("# MedBot System Evaluation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    
    components = ['RAG System', 'BioGPT', 'Clinical-BERT', 'GitHub AI']
    sim_keys = ['rag', 'biogpt', 'clinbert', 'github']
    
    for component, key in zip(components, sim_keys):
        scores = similarities[key]
        report.append(f"### {component}")
        report.append(f"- Average Score: {np.mean(scores):.1f}%")
        report.append(f"- Standard Deviation: {np.std(scores):.1f}%")
        report.append(f"- Min Score: {np.min(scores):.1f}%")
        report.append(f"- Max Score: {np.max(scores):.1f}%")
        report.append("")
    
    # Detailed results
    report.append("## Detailed Question Results")
    report.append("")
    
    for idx, row in results_df.iterrows():
        if row['status'] == 'success':
            report.append(f"### Question {idx+1}")
            report.append(f"**Q:** {row['question']}")
            report.append("")
            report.append(f"**Expected:** {row['expected'][:200]}...")
            report.append("")
            report.append(f"**RAG Answer ({similarities['rag'][idx]:.1f}%):** {row['rag_answer'][:200]}...")
            report.append("")
            report.append(f"**BioGPT Answer ({similarities['biogpt'][idx]:.1f}%):** {row['biogpt_answer'][:200]}...")
            report.append("")
            report.append(f"**Clinical-BERT Answer ({similarities['clinbert'][idx]:.1f}%):** {row['clinbert_answer'][:200]}...")
            report.append("")
            if 'API Error' not in row['github_answer']:
                report.append(f"**GitHub AI Answer ({similarities['github'][idx]:.1f}%):** {row['github_answer'][:200]}...")
            else:
                report.append(f"**GitHub AI Answer:** {row['github_answer']}")
            report.append("")
            report.append("---")
            report.append("")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'MedBot_Evaluation_Report_{timestamp}.md'
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Detailed report saved as: {report_filename}")
    return report_filename

def main():
    """Main evaluation function"""
    print("=" * 80)
    print("MEDBOT SYSTEM EVALUATION")
    print("=" * 80)
    print()
    
    print("📋 This script will:")
    print("1. Load FAQ test questions")
    print("2. Test the MedBot system (make sure app.py is running!)")
    print("3. Calculate semantic similarities")
    print("4. Generate visual results")
    print("5. Create detailed report")
    print()
    
    # Check if Flask app is running
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        print("✅ Flask app is running")
    except:
        print("❌ Flask app is not running!")
        print("   Please start it with: python app.py")
        print("   Then run this script again.")
        return
    
    # Test the system
    results_df = test_rag_system()
    if results_df is None:
        print("❌ Failed to test system")
        return
    
    # Evaluate answers
    similarities = evaluate_answers(results_df)
    
    # Create plots
    plot_filename = create_evaluation_plots(similarities)
    
    # Generate report
    report_filename = generate_detailed_report(results_df, similarities)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"📊 Visual Results: {plot_filename}")
    print(f"📝 Detailed Report: {report_filename}")
    print()
    
    # Print summary
    avg_scores = {
        'RAG System': np.mean(similarities['rag']),
        'BioGPT': np.mean(similarities['biogpt']),
        'Clinical-BERT': np.mean(similarities['clinbert']),
        'GitHub AI': np.mean(similarities['github'])
    }
    
    print("🏆 FINAL SCORES:")
    for component, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {component}: {score:.1f}%")
    
    print()
    print("✅ All files saved in current directory!")

if __name__ == "__main__":
    main()