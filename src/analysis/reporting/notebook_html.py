"""Generate HTML report from analysis results matching the notebook structure."""
import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.core.data_structures import AlignmentResult
from src.analysis.reporting.results import ResultsManager, _get_project_root
from src.analysis.reporting.utils import (
    build_run_id,
    get_layer_label,
    get_summary_columns,
    figure_to_base64,
    extract_layer_number,
)


def generate_notebook_html(
    result_files: List[str],
    output_path: Optional[str] = None,
    include_diagnostic_plots: bool = True,
    include_comparison_plots: bool = True
) -> str:
    """
    Generate an HTML report that matches the structure of analysis.ipynb.
    
    Args:
        result_files: List of paths to result JSON files
        output_path: Optional output path for HTML file. If None, uses 'results/report.html'
        include_diagnostic_plots: Whether to include diagnostic plots (requires plot_diagnostics)
        include_comparison_plots: Whether to include comparison plots when multiple files are provided
        
    Returns:
        Path to generated HTML file
    """
    manager = ResultsManager()
    
    # Determine output path
    if output_path is None:
        project_root = _get_project_root()
        output_path = project_root / "results" / "report.html"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_data = []
    for filepath in result_files:
        metadata, results = manager.load(filepath)
        df = pd.DataFrame([r.__dict__ for r in results])
        all_data.append({
            'metadata': metadata,
            'results': results,
            'df': df,
            'filepath': filepath
        })
    
    # Generate HTML content
    html_content = _generate_html_content(
        all_data, 
        include_diagnostic_plots=include_diagnostic_plots,
        include_comparison_plots=include_comparison_plots and len(result_files) > 1
    )
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML report generated: {output_path}")
    return str(output_path)


def _get_plot_explanations() -> Dict[str, str]:
    """Get explanations for all plot types in the HTML report.
    
    Returns:
        Dictionary mapping plot names to their explanations
    """
    return {
        "Comparison Plots": """
        <strong>Comparison Plots:</strong> This figure contains four subplots comparing multiple analysis runs:
        <ul>
            <li><strong>Alignment vs k:</strong> Shows the mean alignment score (projection energy) as a function of k (number of top singular vectors used). 
                <ul>
                    <li><strong>Formula:</strong> align(k) = Σᵢ₌₁ᵏ (vᵢᵀ · r)², where vᵢ are the top-k right singular vectors from SVD of expert weight matrix, and r is the normalized router vector.</li>
                    <li><strong>Interpretation:</strong> The alignment score is the sum of squared projections of the router vector onto the top-k right singular vectors of the expert weight matrix. For k=1, this equals cos²(θ) between the router vector and the top singular vector. For k>1, it sums the squared projections across multiple singular vectors, measuring how much of the router vector's energy lies in the top-k dimensional subspace of the expert.</li>
                    <li><strong>Computation:</strong> (1) Perform SVD on each expert weight matrix W to get right singular vectors V (columns are singular vectors), (2) Normalize the router vector to unit length, (3) Project the normalized router vector onto the top-k columns of V: proj = V[:, :k]ᵀ @ router_vec, (4) Sum the squared projections: align = Σ(proj²).</li>
                    <li><strong>Range:</strong> [0, 1]. Value of 1 means the router vector lies entirely in the top-k subspace. Value of 0 means it's orthogonal to that subspace.</li>
                    <li><strong>Higher values indicate:</strong> Stronger alignment - the router vector is well-aligned with the principal directions of the expert weight matrix.</li>
                </ul>
            </li>
            <li><strong>Z-score vs k:</strong> Shows the z-score of alignment compared to shuffled baselines.
                <ul>
                    <li><strong>Formula:</strong> z(k) = (align(k) - shuffle_mean(k)) / shuffle_std(k)</li>
                    <li><strong>Interpretation:</strong> Measures how many standard deviations the actual alignment is above the shuffle baseline. This is a normalized measure of statistical significance.</li>
                    <li><strong>Computation:</strong> (1) Compute shuffle_mean and shuffle_std by shuffling router-expert assignments many times (typically 200) and computing alignment for each shuffle, (2) Calculate z-score = (actual_alignment - shuffle_mean) / shuffle_std.</li>
                    <li><strong>Interpretation thresholds:</strong> z > 2 indicates ~95% confidence, z > 3 indicates ~99.7% confidence that alignment exceeds chance. Values near 0 indicate alignment is consistent with random assignments.</li>
                    <li><strong>Higher values indicate:</strong> More statistically significant alignment above the shuffle baseline.</li>
                </ul>
            </li>
            <li><strong>Effect over Random vs k:</strong> Shows the effect size over theoretical random baseline.
                <ul>
                    <li><strong>Formula:</strong> effect_over_random(k) = align(k) - (k / d_model)</li>
                    <li><strong>Interpretation:</strong> Measures how much the actual alignment exceeds the theoretical expectation if the router vector were randomly oriented in d_model-dimensional space. The baseline k/d_model is the expected projection energy onto a random k-dimensional subspace.</li>
                    <li><strong>Computation:</strong> (1) Calculate theoretical baseline: random_expect = k / d_model (where d_model is the model dimension, typically 4096), (2) Subtract from actual alignment: effect = align - random_expect.</li>
                    <li><strong>Baseline explanation:</strong> If a unit vector is randomly oriented in d_model dimensions, the expected projection energy onto any k-dimensional subspace is k/d_model. This is a simple analytical result from random matrix theory.</li>
                    <li><strong>Positive values indicate:</strong> Alignment exceeds theoretical random expectation. Negative values indicate alignment is below even random expectation (rare but possible).</li>
                    <li><strong>Limitation:</strong> This baseline assumes completely random orientation and doesn't account for the actual structure of router and expert vectors.</li>
                </ul>
            </li>
            <li><strong>Delta vs Shuffle vs k:</strong> Shows the difference between actual alignment and empirical shuffle baseline.
                <ul>
                    <li><strong>Formula:</strong> delta(k) = align(k) - shuffle_mean(k)</li>
                    <li><strong>Interpretation:</strong> Measures the raw difference between actual alignment and the empirical mean from shuffled assignments. This preserves the actual structure of router and expert vectors but randomizes which router is assigned to which expert.</li>
                    <li><strong>Computation:</strong> (1) Perform many shuffles (typically 200): randomly permute which router vector is assigned to which expert, (2) For each shuffle, compute alignment using the shuffled assignments, (3) Calculate shuffle_mean = mean of all shuffle alignments, (4) Calculate delta = actual_alignment - shuffle_mean.</li>
                    <li><strong>Why it's more realistic:</strong> Unlike the theoretical baseline, this preserves the actual structure and magnitude of router and expert vectors. It only randomizes the assignment, making it a more appropriate null hypothesis for testing whether specific router-expert pairs are aligned.</li>
                    <li><strong>Positive values indicate:</strong> Actual alignment exceeds the empirical shuffle baseline, suggesting meaningful router-expert alignment beyond random assignment.</li>
                    <li><strong>Comparison to Effect over Random:</strong> Delta vs Shuffle is typically more conservative (smaller values) because shuffle_mean accounts for the actual vector structures, whereas k/d_model assumes completely random vectors.</li>
                </ul>
            </li>
        </ul>
        """,
        "Cos²(θ) Expert Comparison": """
        <strong>Cos²(θ) Expert Comparison:</strong> This figure compares cos²(θ) values across experts and layers (for k=1 only):
        <ul>
            <li><strong>Left plot:</strong> Shows cos²(θ) per expert for k=1 (using only the top singular vector). Each line represents a different layer.
                <ul>
                    <li><strong>Formula:</strong> cos²(θ) = (rᵀ · v₁)², where r is the normalized router vector and v₁ is the top right singular vector (first column of V from SVD).</li>
                    <li><strong>Interpretation:</strong> Measures the squared cosine of the angle between the router vector and the principal direction (top singular vector) of the expert weight matrix. This is a direct correlation measure indicating how well-aligned the router is with the expert's primary direction.</li>
                    <li><strong>Computation:</strong> (1) Perform SVD on expert weight matrix to get V (right singular vectors), (2) Extract v₁ = V[:, 0] (top singular vector), (3) Normalize router vector to unit length, (4) Compute cos²(θ) = (router_vecᵀ · v₁)².</li>
                    <li><strong>Relationship to alignment:</strong> For k=1, cos²(θ) = align(k=1). For k>1, align(k) = Σᵢ₌₁ᵏ cos²(θᵢ) where θᵢ is the angle with the i-th singular vector.</li>
                    <li><strong>Range:</strong> [0, 1]. Value of 1 means router is perfectly aligned with top singular vector. Value of 0 means router is orthogonal to it.</li>
                    <li><strong>Higher values indicate:</strong> Stronger alignment between router and expert's principal direction.</li>
                </ul>
            </li>
            <li><strong>Right plot:</strong> Shows mean cos²(θ) across all experts for each layer at k=1. Bar heights represent the average alignment strength per layer.
                <ul>
                    <li><strong>Formula:</strong> mean_cos²(θ) = (1/n_experts) · Σᵢ cos²(θᵢ), where the sum is over all experts in the layer.</li>
                    <li><strong>Interpretation:</strong> Average alignment strength across all experts in a layer. Provides a layer-level summary of router-expert alignment.</li>
                    <li><strong>Computation:</strong> (1) Compute cos²(θ) for each expert at k=1, (2) Average across all experts in the layer.</li>
                    <li><strong>Use case:</strong> Compare alignment strength across different layers. Higher values indicate stronger overall alignment in that layer.</li>
                </ul>
            </li>
        </ul>
        """,
        "Shuffle Statistics": """
        <strong>Shuffle Statistics:</strong> This figure shows statistics from shuffled baseline comparisons:
        <ul>
            <li><strong>Shuffle Mean vs k:</strong> Mean alignment value from shuffled router-expert assignments as a function of k.
                <ul>
                    <li><strong>Formula:</strong> shuffle_mean(k) = (1/n_shuffles) · Σᵢ align_shuffled_i(k), where align_shuffled_i is the alignment computed with the i-th shuffled assignment.</li>
                    <li><strong>Interpretation:</strong> The expected alignment under the null hypothesis that router-expert assignments are random. This is the empirical baseline used for statistical comparison.</li>
                    <li><strong>Computation:</strong> (1) For each shuffle iteration (typically 200): randomly permute which router vector is assigned to which expert, (2) Compute alignment for each shuffled assignment using the same projection energy formula, (3) Average all shuffle alignments: shuffle_mean = mean(align_shuffled).</li>
                    <li><strong>Why it matters:</strong> This provides the null distribution mean. Actual alignment significantly above this suggests meaningful structure beyond random assignment.</li>
                    <li><strong>Typical behavior:</strong> Usually increases with k (more dimensions = higher projection energy), but typically lower than actual alignment when there's real structure.</li>
                </ul>
            </li>
            <li><strong>Shuffle Std vs k:</strong> Standard deviation of alignment values from shuffled assignments.
                <ul>
                    <li><strong>Formula:</strong> shuffle_std(k) = std(align_shuffled(k)) = √[(1/(n-1)) · Σᵢ (align_shuffled_i - shuffle_mean)²]</li>
                    <li><strong>Interpretation:</strong> Measures the variability in alignment when router-expert assignments are randomized. Larger values indicate more uncertainty in the null distribution.</li>
                    <li><strong>Computation:</strong> (1) Compute alignments for all shuffle iterations, (2) Calculate standard deviation across all shuffle alignments.</li>
                    <li><strong>Use in z-score:</strong> Used as the denominator in z-score calculation: z = (align - shuffle_mean) / shuffle_std. Larger std means smaller z-scores for the same delta, making significance harder to achieve.</li>
                    <li><strong>Typical behavior:</strong> Usually increases with k, as higher-dimensional projections have more variability.</li>
                </ul>
            </li>
            <li><strong>Log Shuffle Std vs k:</strong> Logarithm of shuffle standard deviation.
                <ul>
                    <li><strong>Formula:</strong> log_shuffle_std(k) = log(shuffle_std(k) + ε), where ε is a small constant (typically 1e-10) to avoid log(0).</li>
                    <li><strong>Interpretation:</strong> Logarithmic scale makes it easier to visualize exponential or power-law relationships in the standard deviation.</li>
                    <li><strong>Computation:</strong> (1) Compute shuffle_std as above, (2) Apply natural logarithm: log(std + ε).</li>
                    <li><strong>Why use log scale:</strong> If std grows exponentially with k, the log plot will show a linear relationship, making patterns easier to identify.</li>
                    <li><strong>Use case:</strong> Helps identify whether variability grows exponentially, linearly, or sub-linearly with k.</li>
                </ul>
            </li>
        </ul>
        """,
        "Z-score Decomposition": """
        <strong>Z-score Decomposition:</strong> This figure breaks down the z-score calculation into its components:
        <ul>
            <li><strong>Delta vs k:</strong> The numerator of z-score.
                <ul>
                    <li><strong>Formula:</strong> Δ(k) = align(k) - shuffle_mean(k)</li>
                    <li><strong>Interpretation:</strong> The raw difference between actual alignment and the shuffle baseline. This is the "effect size" before normalization.</li>
                    <li><strong>Computation:</strong> (1) Compute actual alignment for each k, (2) Compute shuffle_mean for each k (from shuffle statistics), (3) Calculate delta = align - shuffle_mean for each k.</li>
                    <li><strong>Units:</strong> Same as alignment (dimensionless, range [0, 1] for alignment, so delta can be negative or positive).</li>
                    <li><strong>Positive values indicate:</strong> Actual alignment exceeds shuffle baseline. Negative values indicate actual alignment is below shuffle baseline (rare but possible).</li>
                    <li><strong>Relationship to z-score:</strong> Delta is the numerator. Larger delta (with same std) leads to larger z-score.</li>
                </ul>
            </li>
            <li><strong>Shuffle Std vs k:</strong> The denominator of z-score.
                <ul>
                    <li><strong>Formula:</strong> σ_shuffle(k) = std(align_shuffled(k))</li>
                    <li><strong>Interpretation:</strong> The variability in shuffled alignments. This is the same metric shown in Shuffle Statistics plot, but displayed here to show its role in z-score normalization.</li>
                    <li><strong>Computation:</strong> Standard deviation across all shuffle iterations for each k value. Same as described in Shuffle Statistics.</li>
                    <li><strong>Role in z-score:</strong> Acts as the normalization factor. Larger std means the same delta produces a smaller z-score, making it harder to achieve statistical significance.</li>
                    <li><strong>Why it matters:</strong> Understanding std helps interpret z-scores. A large delta with large std might have a moderate z-score, while a smaller delta with small std might have a large z-score.</li>
                </ul>
            </li>
            <li><strong>Z-score vs k:</strong> The final z-score.
                <ul>
                    <li><strong>Formula:</strong> z(k) = Δ(k) / σ_shuffle(k) = (align(k) - shuffle_mean(k)) / shuffle_std(k)</li>
                    <li><strong>Interpretation:</strong> The number of standard deviations the actual alignment is above (or below) the shuffle baseline. This is a normalized measure of statistical significance.</li>
                    <li><strong>Computation:</strong> (1) Compute delta for each k, (2) Compute shuffle_std for each k, (3) Calculate z = delta / shuffle_std for each k.</li>
                    <li><strong>Statistical interpretation:</strong> Under the null hypothesis (random assignments), z follows approximately a standard normal distribution. z > 2 indicates ~95% confidence (p < 0.05), z > 3 indicates ~99.7% confidence (p < 0.003) that alignment exceeds chance.</li>
                    <li><strong>Advantages over delta:</strong> Normalized measure that accounts for variability. A delta of 0.1 might be significant if std=0.02 (z=5) but not if std=0.1 (z=1).</li>
                    <li><strong>Higher values indicate:</strong> More statistically significant alignment above the shuffle baseline.</li>
                </ul>
            </li>
        </ul>
        """,
        "Distribution Comparison": """
        <strong>Distribution Comparison:</strong> This plot shows the probability distribution of shuffled alignments (projection energies) compared to the true alignment value:
        <ul>
            <li><strong>Solid curves:</strong> Approximate normal distribution of alignment values (projection energies) from shuffled router-expert assignments.
                <ul>
                    <li><strong>Formula:</strong> P(align) ≈ N(μ_shuffle, σ²_shuffle), where μ_shuffle = shuffle_mean and σ_shuffle = shuffle_std.</li>
                    <li><strong>Interpretation:</strong> The probability distribution of what alignment values we would expect by chance when router vectors are randomly assigned to experts. This is the null distribution for statistical testing.</li>
                    <li><strong>Computation:</strong> (1) Compute shuffle_mean and shuffle_std from shuffle experiments, (2) Approximate the distribution as a normal distribution: N(shuffle_mean, shuffle_std²), (3) Plot the probability density function using scipy.stats.norm.pdf(x, shuffle_mean, shuffle_std).</li>
                    <li><strong>Why normal distribution:</strong> By the Central Limit Theorem, the mean of many independent shuffle alignments approximates a normal distribution, especially with 200 shuffles.</li>
                    <li><strong>What it shows:</strong> The range and likelihood of alignment values under the null hypothesis. The peak is at shuffle_mean, and the width is determined by shuffle_std.</li>
                </ul>
            </li>
            <li><strong>Dashed vertical lines:</strong> The actual alignment value (projection energy) for each run.
                <ul>
                    <li><strong>Formula:</strong> true_align(k) = (1/n_experts) · Σᵢ align_i(k), where align_i is the alignment for expert i at k.</li>
                    <li><strong>Interpretation:</strong> The observed mean alignment across all experts for the given k value. This is what we're testing against the null distribution.</li>
                    <li><strong>Computation:</strong> (1) Compute alignment for each expert at the given k value, (2) Average across all experts: true_align = mean(align_expert).</li>
                    <li><strong>Position relative to distribution:</strong> If the line is far to the right of the distribution peak (shuffle_mean), it indicates strong alignment above chance. The distance from the peak, measured in standard deviations, corresponds to the z-score.</li>
                    <li><strong>Statistical interpretation:</strong> If the line falls in the right tail of the distribution (beyond ~2σ), it suggests the alignment is statistically significant (p < 0.05).</li>
                    <li><strong>Multiple runs:</strong> Each run gets its own dashed line, allowing comparison of alignment strength across different layers or configurations.</li>
                </ul>
            </li>
        </ul>
        """,
        "Per-Expert Breakdown": """
        <strong>Per-Expert Breakdown:</strong> This figure provides detailed expert-level analysis:
        <ul>
            <li><strong>Alignment Heatmap (top left):</strong> Shows alignment values (projection energies) for each expert (rows) across different k values (columns).
                <ul>
                    <li><strong>Formula:</strong> heatmap[expert, k] = mean(align_expert,k), averaged across any multiple runs if present.</li>
                    <li><strong>Interpretation:</strong> Visual representation of how alignment strength varies across experts and k values. Each cell shows the mean alignment for a specific expert-k combination.</li>
                    <li><strong>Computation:</strong> (1) Group data by expert and k, (2) Average alignment values within each group, (3) Create pivot table: pivot = df.pivot_table(values='align', index='expert', columns='k', aggfunc='mean'), (4) Display as heatmap with color intensity proportional to alignment value.</li>
                    <li><strong>Color scheme:</strong> Warmer colors (yellow/green) indicate stronger alignment, cooler colors (blue/purple) indicate weaker alignment. Uses 'viridis' colormap.</li>
                    <li><strong>What to look for:</strong> Patterns across experts (rows) show which experts have consistently high/low alignment. Patterns across k (columns) show how alignment changes with dimensionality.</li>
                    <li><strong>Use case:</strong> Identify experts with particularly strong or weak alignment, and see how alignment scales with k for each expert.</li>
                </ul>
            </li>
            <li><strong>Delta Heatmap (top right):</strong> Shows delta (alignment - shuffle_mean) for each expert across k values.
                <ul>
                    <li><strong>Formula:</strong> heatmap[expert, k] = mean(align_expert,k - shuffle_mean_k), averaged across runs if present.</li>
                    <li><strong>Interpretation:</strong> Shows how much each expert's alignment exceeds (or falls below) the shuffle baseline at each k value.</li>
                    <li><strong>Computation:</strong> (1) Compute delta for each expert-k combination: delta = align - shuffle_mean, (2) Create pivot table: pivot = df.pivot_table(values='delta_vs_shuffle', index='expert', columns='k', aggfunc='mean'), (3) Display as heatmap with colormap centered at zero.</li>
                    <li><strong>Color scheme:</strong> Red indicates positive delta (above shuffle baseline), blue indicates negative delta (below shuffle baseline). Uses 'RdBu_r' (Red-Blue reversed) colormap, centered at zero using vmin=-max_abs, vmax=max_abs.</li>
                    <li><strong>What to look for:</strong> Experts with consistently red cells have strong alignment above baseline. Experts with blue cells have alignment below baseline (rare but possible).</li>
                    <li><strong>Advantage over alignment heatmap:</strong> Normalized by shuffle baseline, making it easier to see which experts truly exceed chance expectations.</li>
                </ul>
            </li>
            <li><strong>Alignment vs Expert (bottom left):</strong> Scatter plot showing alignment values (projection energies) for each expert at a fixed k (typically k=128).
                <ul>
                    <li><strong>Formula:</strong> For each expert i: align_i(k_fixed), where k_fixed is typically 128 or the median k value if 128 is not available.</li>
                    <li><strong>Interpretation:</strong> Shows the distribution of alignment strengths across experts at a representative k value. Each point represents one expert's alignment.</li>
                    <li><strong>Computation:</strong> (1) Select a representative k value (prefer k=128, fallback to median k), (2) Filter data: k_data = df[df['k'] == k_fixed], (3) Extract alignment values: align_vals = k_data['align'].values, expert_vals = k_data['expert'].values, (4) Plot as scatter: scatter(expert_vals, align_vals).</li>
                    <li><strong>Why scatter plot:</strong> Shows individual expert values rather than averages, revealing variability and outliers.</li>
                    <li><strong>What to look for:</strong> Experts with particularly high or low alignment values. Clustering of points suggests similar alignment strengths across experts.</li>
                    <li><strong>Multiple runs:</strong> If comparing multiple runs, each run gets a different color/marker, allowing comparison of alignment patterns across layers or configurations.</li>
                </ul>
            </li>
            <li><strong>Delta vs Expert (bottom right):</strong> Scatter plot showing delta values for each expert at a fixed k.
                <ul>
                    <li><strong>Formula:</strong> For each expert i: delta_i(k_fixed) = align_i(k_fixed) - shuffle_mean(k_fixed).</li>
                    <li><strong>Interpretation:</strong> Shows how much each expert's alignment exceeds (or falls below) the shuffle baseline at a representative k value.</li>
                    <li><strong>Computation:</strong> (1) Use the same k_fixed as in Alignment vs Expert plot, (2) Filter data: k_data = df[df['k'] == k_fixed], (3) Extract delta values: delta_vals = k_data['delta_vs_shuffle'].values, expert_vals = k_data['expert'].values, (4) Plot as scatter: scatter(expert_vals, delta_vals), (5) Add horizontal line at y=0 for reference.</li>
                    <li><strong>Reference line:</strong> The horizontal dashed line at y=0 separates experts above baseline (positive delta) from those below baseline (negative delta).</li>
                    <li><strong>What to look for:</strong> Experts with delta significantly above zero have strong alignment. Most experts should have positive delta if there's meaningful structure. Negative delta is rare but indicates alignment below even random assignment.</li>
                    <li><strong>Advantage over Alignment vs Expert:</strong> Normalized by shuffle baseline, making it easier to identify experts with statistically meaningful alignment.</li>
                    <li><strong>Multiple runs:</strong> If comparing multiple runs, each run gets different markers, showing how delta patterns vary across layers or configurations.</li>
                </ul>
            </li>
        </ul>
        """,
        "Complete Analysis Visualization": """
        <strong>Complete Analysis Visualization:</strong> This comprehensive figure contains 12 subplots showing all key metrics for a single analysis run:
        <ul>
            <li><strong>Row 1:</strong> Alignment, Z-score, and Effect over Random vs k (log scale)</li>
            <li><strong>Row 2:</strong> Delta vs Shuffle, Shuffle Mean, and Shuffle Std vs k</li>
            <li><strong>Row 3:</strong> Heatmaps showing Alignment, Delta, and Z-score across experts (rows) and k values (columns)</li>
            <li><strong>Row 4:</strong> Cos²(θ) per expert (k=1), Alignment per expert, and Z-score per expert at a representative k value</li>
        </ul>
        All metrics are computed as described in the individual plot explanations above.
        """
    }


def _generate_comparison_plots(all_data: List[dict]) -> Dict[str, str]:
    """Generate comparison plots for multiple result files.
    
    Returns:
        Dictionary mapping plot names to base64-encoded image data
    """
    plots = {}
    
    # Prepare combined data
    combined_data = []
    for data in all_data:
        metadata = data['metadata']
        df = data['df']
        
        summary_cols = get_summary_columns(df)
        summary = df.groupby("k")[summary_cols].mean().reset_index()
        # Use layer-only label for legends
        layer_label = get_layer_label(metadata)
        summary["run"] = layer_label
        combined_data.append(summary)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    runs = sorted(combined_df["run"].unique(), key=extract_layer_number)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comparison Across All Runs", fontsize=16, fontweight='bold')
    
    # Plot 1: Alignment vs k
    ax = axes[0, 0]
    for run in runs:
        run_data = combined_df[combined_df["run"] == run]
        ax.plot(
            run_data["k"], 
            run_data["align"], 
            marker='o', 
            label=run,
            linewidth=2,
            markersize=6
        )
    ax.set_xlabel('k (number of singular vectors)', fontsize=11)
    ax.set_ylabel('Mean Alignment', fontsize=11)
    ax.set_title('Alignment vs k', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.legend(loc='best', fontsize=9)
    
    # Plot 2: Z-score vs k
    ax = axes[0, 1]
    for run in runs:
        run_data = combined_df[combined_df["run"] == run]
        ax.plot(
            run_data["k"], 
            run_data["z_vs_shuffle"], 
            marker='s', 
            label=run,
            linewidth=2,
            markersize=6
        )
    ax.set_xlabel('k (number of singular vectors)', fontsize=11)
    ax.set_ylabel('Mean Z-score', fontsize=11)
    ax.set_title('Z-score vs k', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    
    # Plot 3: Effect over random vs k
    ax = axes[1, 0]
    for run in runs:
        run_data = combined_df[combined_df["run"] == run]
        ax.plot(
            run_data["k"], 
            run_data["effect_over_random"], 
            marker='^', 
            label=run,
            linewidth=2,
            markersize=6
        )
    ax.set_xlabel('k (number of singular vectors)', fontsize=11)
    ax.set_ylabel('Mean Effect over Random', fontsize=11)
    ax.set_title('Effect over Random vs k', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    
    # Plot 4: Delta vs shuffle vs k
    ax = axes[1, 1]
    for run in runs:
        run_data = combined_df[combined_df["run"] == run]
        ax.plot(
            run_data["k"], 
            run_data["delta_vs_shuffle"], 
            marker='d', 
            label=run,
            linewidth=2,
            markersize=6
        )
    ax.set_xlabel('k (number of singular vectors)', fontsize=11)
    ax.set_ylabel('Mean Delta vs Shuffle', fontsize=11)
    ax.set_title('Delta vs Shuffle vs k', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    
    # Apply tight layout before saving
    try:
        plt.tight_layout()
    except Exception:
        pass  # Some axes may not be compatible with tight_layout
    
    # Convert to base64
    img_data = figure_to_base64(fig)
    plt.close(fig)
    
    plots["Comparison Plots"] = img_data
    
    # Add expert comparison plots if we have cos_squared data
    has_cos_squared = False
    for data in all_data:
        df = data['df']
        if 1 in df['k'].values and 'cos_squared' in df.columns:
            has_cos_squared = True
            break
    
    if has_cos_squared and len(all_data) > 1:
        # Cos²(θ) per expert across runs
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Cos²(θ) Expert Comparison (k=1)', fontsize=14, fontweight='bold')
        
        # Left plot: Cos²(θ) per expert
        ax = axes[0]
        # Sort all_data by layer number
        sorted_all_data = sorted(all_data, key=lambda d: extract_layer_number(get_layer_label(d['metadata'])))
        for data in sorted_all_data:
            metadata = data['metadata']
            df = data['df']
            layer_label = get_layer_label(metadata)
            
            k1_expert = df[(df['k'] == 1) & (df['cos_squared'].notna())].sort_values('expert')
            if len(k1_expert) > 0:
                ax.plot(k1_expert['expert'], k1_expert['cos_squared'], 
                       marker='o', label=layer_label, linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_xlabel('Expert Index', fontsize=11)
        ax.set_ylabel('Cos²(θ)', fontsize=11)
        ax.set_title('Cos²(θ) per Expert Across Layers (k=1)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if len(ax.get_xticks()) > 0:
            max_expert = int(df['expert'].max()) if len(df) > 0 else 7
            ax.set_xticks(range(max_expert + 1))
        
        # Right plot: Mean Cos²(θ) comparison
        ax = axes[1]
        cos_stats = []
        # Sort all_data by layer number
        sorted_all_data = sorted(all_data, key=lambda d: extract_layer_number(get_layer_label(d['metadata'])))
        for data in sorted_all_data:
            metadata = data['metadata']
            df = data['df']
            layer_label = get_layer_label(metadata)
            k1_data = df[(df['k'] == 1) & (df['cos_squared'].notna())]
            if len(k1_data) > 0:
                mean_cos = k1_data['cos_squared'].mean()
                cos_stats.append({'run': layer_label, 'cos_squared': mean_cos})
        
        if cos_stats:
            cos_df = pd.DataFrame(cos_stats)
            bars = ax.bar(range(len(cos_df)), cos_df['cos_squared'], alpha=0.7)
            ax.set_xticks(range(len(cos_df)))
            ax.set_xticklabels(cos_df['run'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Mean Cos²(θ)', fontsize=11)
            ax.set_title('Mean Cos²(θ) Comparison (k=1)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, cos_df['cos_squared'])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plots['Cos²(θ) Expert Comparison'] = figure_to_base64(fig)
        plt.close(fig)
    
    return plots


def _generate_diagnostic_plots(all_data: List[dict], layer) -> Dict[str, str]:
    """Generate diagnostic plots for HTML embedding.
    
    Returns:
        Dictionary mapping plot names to base64-encoded image data
    """
    plots = {}
    
    try:
        import numpy as np
        from scipy import stats
        
        # Prepare data similar to ResultsManager._plot_diagnostics
        diagnostic_data = []
        for data in all_data:
            metadata = data['metadata']
            df = data['df'].copy()
            # Use layer-only label for legends
            layer_label = get_layer_label(metadata)
            df['run_id'] = layer_label
            diagnostic_data.append({
                'metadata': metadata,
                'df': df,
                'filepath': data['filepath']
            })
        
        # Sort diagnostic_data by layer number
        diagnostic_data = sorted(diagnostic_data, key=lambda d: extract_layer_number(get_layer_label(d['metadata'])))
        
        # 1. Shuffle statistics
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Shuffle Statistics vs k (Layer {layer})', fontsize=14, fontweight='bold')
        
        for data in diagnostic_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            
            # Mean shuffle vs k
            mean_by_k = df.groupby('k')['shuffle_mean'].mean()
            axes[0].plot(mean_by_k.index, mean_by_k.values, marker='o', label=run_id, linewidth=2)
            
            # Std shuffle vs k
            std_by_k = df.groupby('k')['shuffle_std'].mean()
            axes[1].plot(std_by_k.index, std_by_k.values, marker='s', label=run_id, linewidth=2)
            
            # Log(std) shuffle vs k
            log_std = np.log(std_by_k.values + 1e-10)
            axes[2].plot(std_by_k.index, log_std, marker='^', label=run_id, linewidth=2)
        
        axes[0].set_xlabel('k', fontsize=11)
        axes[0].set_ylabel('μ_shuffle(k)', fontsize=11)
        axes[0].set_title('Shuffle Mean vs k', fontsize=12)
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)
        
        axes[1].set_xlabel('k', fontsize=11)
        axes[1].set_ylabel('σ_shuffle(k)', fontsize=11)
        axes[1].set_title('Shuffle Std vs k', fontsize=12)
        axes[1].set_xscale('log', base=2)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
        
        axes[2].set_xlabel('k', fontsize=11)
        axes[2].set_ylabel('log(σ_shuffle(k))', fontsize=11)
        axes[2].set_title('Log Shuffle Std vs k', fontsize=12)
        axes[2].set_xscale('log', base=2)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=9)
        
        plt.tight_layout()
        plots['Shuffle Statistics'] = figure_to_base64(fig)
        plt.close(fig)
        
        # 2. Z-score decomposition
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Z-score Decomposition vs k (Layer {layer})', fontsize=14, fontweight='bold')
        
        for data in diagnostic_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            
            delta_by_k = df.groupby('k')['delta_vs_shuffle'].mean()
            axes[0].plot(delta_by_k.index, delta_by_k.values, marker='o', label=run_id, linewidth=2)
            
            std_by_k = df.groupby('k')['shuffle_std'].mean()
            axes[1].plot(std_by_k.index, std_by_k.values, marker='s', label=run_id, linewidth=2)
            
            z_by_k = df.groupby('k')['z_vs_shuffle'].mean()
            axes[2].plot(z_by_k.index, z_by_k.values, marker='^', label=run_id, linewidth=2)
        
        axes[0].set_xlabel('k', fontsize=11)
        axes[0].set_ylabel('Δ(k) = align(k) - μ_shuffle(k)', fontsize=11)
        axes[0].set_title('Delta vs k', fontsize=12)
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].legend(fontsize=9)
        
        axes[1].set_xlabel('k', fontsize=11)
        axes[1].set_ylabel('σ_shuffle(k)', fontsize=11)
        axes[1].set_title('Shuffle Std vs k', fontsize=12)
        axes[1].set_xscale('log', base=2)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
        
        axes[2].set_xlabel('k', fontsize=11)
        axes[2].set_ylabel('z(k) = Δ(k) / σ_shuffle(k)', fontsize=11)
        axes[2].set_title('Z-score vs k', fontsize=12)
        axes[2].set_xscale('log', base=2)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].legend(fontsize=9)
        
        plt.tight_layout()
        plots['Z-score Decomposition'] = figure_to_base64(fig)
        plt.close(fig)
        
        # 3. Distribution plots (for representative k values)
        try:
            all_k_values = set()
            for data in diagnostic_data:
                all_k_values.update(data['df']['k'].unique())
            
            # Select representative k values that exist
            preferred_k = [32, 128, 512, 2048]
            k_values = [k for k in preferred_k if k in all_k_values]
            if not k_values and all_k_values:
                # Fallback to median k values
                sorted_k = sorted(all_k_values)
                k_values = [sorted_k[len(sorted_k) // 4], sorted_k[len(sorted_k) // 2], sorted_k[3 * len(sorted_k) // 4]]
            
            for k in k_values[:4]:  # Limit to 4 plots
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    # Determine if we're comparing multiple layers
                    layers = set(d['metadata'].get('layer', '?') for d in diagnostic_data)
                    if len(layers) > 1:
                        layer_str = f"Layers {sorted(layers)}"
                    else:
                        layer_str = f"Layer {layer}"
                    fig.suptitle(f'Shuffle Distribution vs True Alignment (k={k}, {layer_str}) - Diagnostic', 
                                fontsize=14, fontweight='bold')
                    
                    has_data = False
                    for data in diagnostic_data:
                        df = data['df']
                        run_id = df['run_id'].iloc[0]
                        k_data = df[df['k'] == k]
                        
                        if len(k_data) == 0:
                            continue
                        
                        try:
                            shuffle_mean = k_data['shuffle_mean'].iloc[0]
                            shuffle_std = k_data['shuffle_std'].iloc[0]
                            true_align = k_data['align'].mean()
                            
                            x = np.linspace(max(0, shuffle_mean - 4*shuffle_std), 
                                           shuffle_mean + 4*shuffle_std, 200)
                            y = stats.norm.pdf(x, shuffle_mean, shuffle_std)
                            
                            ax.plot(x, y, label=f'{run_id} (shuffle)', linewidth=2, alpha=0.7)
                            ax.axvline(true_align, color=ax.lines[-1].get_color(), 
                                      linestyle='--', linewidth=2, 
                                      label=f'{run_id} (true align)', alpha=0.8)
                            has_data = True
                        except Exception as e:
                            print(f"Warning: Could not plot distribution for {run_id} at k={k}: {e}")
                            continue
                    
                    if has_data:
                        ax.set_xlabel('Alignment', fontsize=11)
                        ax.set_ylabel('Density (approximated)', fontsize=11)
                        ax.set_title(f'Distribution Comparison (k={k})', fontsize=12)
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=9)
                        
                        plt.tight_layout()
                        plots[f'Distribution Comparison (k={k})'] = figure_to_base64(fig)
                        plt.close(fig)
                    else:
                        plt.close(fig)
                except Exception as e:
                    print(f"Warning: Could not generate distribution plot for k={k}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        except Exception as e:
            print(f"Warning: Could not generate distribution plots: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Per-expert breakdown
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Per-Expert Breakdown (Layer {layer}) - Diagnostic', fontsize=14, fontweight='bold')
        
        # Heatmap 1: Alignment per expert vs k
        # Use first run for heatmap (heatmaps don't work well with multiple overlays)
        ax1 = fig.add_subplot(gs[0, 0])
        if len(diagnostic_data) > 0:
            df = diagnostic_data[0]['df']
            run_id = df['run_id'].iloc[0]
            
            pivot = df.pivot_table(values='align', index='expert', columns='k', aggfunc='mean')
            k_values = sorted(pivot.columns)
            
            im = ax1.imshow(pivot.values, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax1, label='Alignment')
        
        ax1.set_xlabel('k', fontsize=11)
        ax1.set_ylabel('Expert', fontsize=11)
        ax1.set_title(f'Alignment Heatmap (Experts × k) - {run_id if len(diagnostic_data) > 0 else ""}', fontsize=12)
        if len(diagnostic_data) > 0:
            ax1.set_xticks(range(len(k_values)))
            ax1.set_xticklabels([str(k) for k in k_values], rotation=45)
            ax1.set_yticks(range(len(pivot.index)))
            ax1.set_yticklabels([int(e) for e in pivot.index])
        
        # Heatmap 2: Delta per expert vs k
        ax2 = fig.add_subplot(gs[0, 1])
        if len(diagnostic_data) > 0:
            df = diagnostic_data[0]['df']
            run_id = df['run_id'].iloc[0]
            
            pivot = df.pivot_table(values='delta_vs_shuffle', index='expert', columns='k', aggfunc='mean')
            # Center colormap at zero for delta
            max_abs = pivot.values.max() if len(pivot.values) > 0 else 1.0
            min_abs = abs(pivot.values.min()) if len(pivot.values) > 0 else 1.0
            vmax = max(max_abs, min_abs)
            
            im = ax2.imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax2, label='Delta')
        
        ax2.set_xlabel('k', fontsize=11)
        ax2.set_ylabel('Expert', fontsize=11)
        ax2.set_title(f'Delta Heatmap (Experts × k) - {run_id if len(diagnostic_data) > 0 else ""}', fontsize=12)
        if len(diagnostic_data) > 0:
            ax2.set_xticks(range(len(k_values)))
            ax2.set_xticklabels([str(k) for k in k_values], rotation=45)
            ax2.set_yticks(range(len(pivot.index)))
            ax2.set_yticklabels([int(e) for e in pivot.index])
        
        # Scatter: Alignment vs expert (for k=128 or median k)
        ax3 = fig.add_subplot(gs[1, 0])
        all_k_vals = set()
        for data in diagnostic_data:
            all_k_vals.update(data['df']['k'].unique())
        k_fixed = 128 if 128 in all_k_vals else sorted(all_k_vals)[len(all_k_vals) // 2]
        
        for data in diagnostic_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            k_data = df[df['k'] == k_fixed]
            
            if len(k_data) > 0:
                align_vals = k_data['align'].values
                expert_vals = k_data['expert'].values
                ax3.scatter(expert_vals, align_vals, label=run_id, alpha=0.7, s=100)
        
        ax3.set_xlabel('Expert Index', fontsize=11)
        ax3.set_ylabel(f'Alignment (k={k_fixed})', fontsize=11)
        ax3.set_title(f'Alignment vs Expert (k={k_fixed})', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # Scatter: Delta vs expert
        ax4 = fig.add_subplot(gs[1, 1])
        for data in diagnostic_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            k_data = df[df['k'] == k_fixed]
            
            if len(k_data) > 0:
                delta_vals = k_data['delta_vs_shuffle'].values
                expert_vals = k_data['expert'].values
                ax4.scatter(expert_vals, delta_vals, label=run_id, alpha=0.7, s=100)
        
        ax4.set_xlabel('Expert Index', fontsize=11)
        ax4.set_ylabel(f'Delta (k={k_fixed})', fontsize=11)
        ax4.set_title(f'Delta vs Expert (k={k_fixed})', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.legend(fontsize=9)
        
        plt.tight_layout()
        plots['Per-Expert Breakdown'] = figure_to_base64(fig)
        plt.close(fig)
        
        # Note: Cos²(θ) Expert Comparison is generated in _generate_comparison_plots
        # to avoid duplication
        
    except Exception as e:
        print(f"Warning: Could not generate all diagnostic plots: {e}")
        import traceback
        traceback.print_exc()
    
    return plots


def _generate_html_content(
    all_data: List[dict], 
    include_diagnostic_plots: bool,
    include_comparison_plots: bool = False
) -> str:
    """Generate HTML content matching notebook structure."""
    # Get plot explanations
    plot_explanations = _get_plot_explanations()
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Router-Expert Alignment Analysis Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 40px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }
        h3 {
            color: #666;
            margin-top: 30px;
        }
        .section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metadata {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .metadata-item {
            margin: 5px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .plot-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: block;
            margin: 20px auto;
        }
        .plot-explanation {
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            text-align: left;
            font-size: 0.95em;
        }
        .plot-explanation ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .plot-explanation ul ul {
            margin: 5px 0;
            padding-left: 20px;
        }
        .plot-explanation li {
            margin: 5px 0;
        }
        .plot-explanation strong {
            color: #4CAF50;
        }
        .code-block {
            background: #f4f4f4;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
        .output {
            background: #f9f9f9;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 15px 0;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
        }
        .cos-squared-section {
            background: #fff3e0;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Router-Expert Alignment Analysis</h1>
    <p>This report performs SVD-based alignment analysis between router vectors and expert weight matrices.</p>
    
    <div class="section">
        <h2>Plot Explanations</h2>
        <p>The following section explains what each plot type shows and how it is computed. All plots use layer numbers (e.g., L5, L10) in their legends, without timestamps.</p>
"""
    
    # Add explanations for all plot types
    for plot_name, explanation in plot_explanations.items():
        html += f"""
        <div style="margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #4CAF50;">
            <h3>{plot_name}</h3>
            {explanation}
        </div>
"""
    
    html += """
    </div>
"""
    
    # Add comparison section FIRST if multiple files
    if include_comparison_plots and len(all_data) > 1:
        html += f"""
    <div class="section">
        <h2>1. Comparison Across All Runs</h2>
        <p>This section compares all {len(all_data)} result files side by side.</p>
"""
        
        # Generate comparison plots
        try:
            comparison_plots = _generate_comparison_plots(all_data)
            for plot_name, plot_data in comparison_plots.items():
                html += f"""
        <div class="plot-container">
            <h4>{plot_name}</h4>
            <p><em>See "Plot Explanations" section at the top of this report for detailed information about this plot.</em></p>
            <img src="data:image/png;base64,{plot_data}" alt="{plot_name}">
        </div>
"""
        except Exception as e:
            html += f"""
        <div class="code-block">
            <p>⚠️ Could not generate comparison plots: {e}</p>
        </div>
"""
        
        # Add diagnostic plots to comparison section
        if include_diagnostic_plots:
            html += """
        <h3>Diagnostic Plots (Comparison)</h3>
        <p>Diagnostic plots comparing all runs to understand differences in alignment, z-score, and delta.</p>
"""
            try:
                # Use first layer for diagnostic plots (or determine from data)
                layer = all_data[0]['metadata'].get('layer', 'unknown')
                diagnostic_plots = _generate_diagnostic_plots(all_data, layer)
                
                if diagnostic_plots:
                    for plot_name, plot_data_b64 in diagnostic_plots.items():
                        # Handle dynamic plot names like "Distribution Comparison (k=128)"
                        if plot_name.startswith("Distribution Comparison"):
                            explanation = plot_explanations.get("Distribution Comparison", "")
                        else:
                            explanation = plot_explanations.get(plot_name, "")
                        html += f"""
        <div class="plot-container">
            <h4>{plot_name}</h4>
            <p><em>See "Plot Explanations" section at the top of this report for detailed information about this plot.</em></p>
            <img src="data:image/png;base64,{plot_data_b64}" alt="{plot_name}">
        </div>
"""
                else:
                    html += """
        <div class="code-block">
            <p>⚠️ Could not generate diagnostic plots. Check data format.</p>
        </div>
"""
            except Exception as e:
                html += f"""
        <div class="code-block">
            <p>⚠️ Could not generate diagnostic plots: {e}</p>
        </div>
"""
        
        html += "</div>\n"
    
    # Process each result file (Individual Analysis)
    for idx, data in enumerate(all_data):
        metadata = data['metadata']
        df = data['df']
        layer = metadata.get('layer', 'unknown')
        
        html += f"""
    <div class="section">
        <h2>{idx + 2 if (include_comparison_plots and len(all_data) > 1) else idx + 1}. Individual Analysis - Run {idx + 1}</h2>
        
        <h3>Setup and Configuration</h3>
        <div class="metadata">
            <div class="metadata-item"><strong>Model:</strong> {metadata.get('model_id', 'unknown')}</div>
            <div class="metadata-item"><strong>Layer:</strong> {layer}</div>
            <div class="metadata-item"><strong>Method:</strong> {metadata.get('method', 'unknown')}</div>
            <div class="metadata-item"><strong>K values:</strong> {metadata.get('k_list', 'unknown')}</div>
            <div class="metadata-item"><strong>Shuffles:</strong> {metadata.get('n_shuffles', 'unknown')}</div>
            <div class="metadata-item"><strong>Seed:</strong> {metadata.get('seed', 'unknown')}</div>
            <div class="metadata-item"><strong>Number of experts:</strong> {metadata.get('n_experts', 'unknown')}</div>
            <div class="metadata-item"><strong>Timestamp:</strong> {metadata.get('datetime', metadata.get('timestamp', 'unknown'))}</div>
        </div>
        
        <h3>Summary Statistics (averaged across experts)</h3>
        <div class="code-block">
"""
        
        # Summary table
        summary_cols = get_summary_columns(df)
        summary = df.groupby("k")[summary_cols].mean()
        
        html += "<table>\n<thead>\n<tr>\n<th>k</th>\n"
        for col in summary_cols:
            html += f"<th>{col}</th>\n"
        html += "</tr>\n</thead>\n<tbody>\n"
        
        for k, row in summary.iterrows():
            html += f"<tr>\n<td>{k}</td>\n"
            for col in summary_cols:
                html += f"<td>{row[col]:.6f}</td>\n"
            html += "</tr>\n"
        
        html += "</tbody>\n</table>\n</div>\n"
        
        # Cos²(θ) section if k=1 exists
        if 1 in df["k"].values:
            k1_results = df[df["k"] == 1]
            html += """
        <h3>Cos²(θ) Alignment (k=1)</h3>
        <div class="cos-squared-section">
"""
            html += f"""
            <p><strong>Mean cos²(θ):</strong> {k1_results['cos_squared'].mean():.6f}</p>
            <p><strong>Max cos²(θ):</strong> {k1_results['cos_squared'].max():.6f}</p>
            <p><strong>Min cos²(θ):</strong> {k1_results['cos_squared'].min():.6f}</p>
            <p><strong>Std cos²(θ):</strong> {k1_results['cos_squared'].std():.6f}</p>
            
            <h4>Per-expert cos²(θ) values:</h4>
            <table>
                <thead>
                    <tr><th>Expert</th><th>cos²(θ)</th><th>align</th></tr>
                </thead>
                <tbody>
"""
            for _, row in k1_results.iterrows():
                html += f"""
                    <tr>
                        <td>{int(row['expert'])}</td>
                        <td>{row['cos_squared']:.6f}</td>
                        <td>{row['align']:.6f}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Detailed results by K value
        html += """
        <h3>Detailed Results by K Value</h3>
        <div class="code-block">
"""
        for k in sorted(df["k"].unique()):
            k_data = df[df["k"] == k]
            html += f"""
            <p><strong>K = {k}:</strong></p>
            <ul>
                <li>Mean align: {k_data['align'].mean():.6f}</li>
                <li>Mean z-score: {k_data['z_vs_shuffle'].mean():.2f}</li>
                <li>Mean effect over random: {k_data['effect_over_random'].mean():.6f}</li>
            </ul>
"""
        html += "</div>\n"
        
        # Single file comprehensive plots
        html += """
        <h3>Complete Analysis Plots</h3>
        <p>Comprehensive visualization of all metrics for this run.</p>
"""
        try:
            from src.analysis.reporting.single_file_plots import generate_single_file_plots
            
            plots = generate_single_file_plots(data['filepath'], save_to_file=False)
            if "complete_analysis" in plots:
                plot_bytes = plots["complete_analysis"]
                plot_b64 = base64.b64encode(plot_bytes).decode('utf-8')
                explanation = plot_explanations.get("Complete Analysis Visualization", "")
                html += f"""
        <div class="plot-container">
            <h4>Complete Analysis Visualization</h4>
            <p><em>See "Plot Explanations" section at the top of this report for detailed information about this plot.</em></p>
            <img src="data:image/png;base64,{plot_b64}" alt="Complete Analysis">
        </div>
"""
        except Exception as e:
            html += f"""
        <div class="code-block">
            <p>⚠️ Could not generate complete analysis plot: {e}</p>
        </div>
"""
        
        # Note: Diagnostic plots are shown in the comparison section above
        # Individual file analysis focuses on the single file's comprehensive plots
        
        html += "</div>\n"  # Close section
    
    html += f"""
    <hr>
    <p style="text-align: center; color: #666; margin-top: 40px;">
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</body>
</html>
"""
    
    return html


def generate_from_notebook(result_file: str, output_path: Optional[str] = None) -> str:
    """
    Convenience function to generate HTML report from a single result file.
    
    Args:
        result_file: Path to result JSON file
        output_path: Optional output path for HTML file
        
    Returns:
        Path to generated HTML file
    """
    return generate_notebook_html([result_file], output_path=output_path)
