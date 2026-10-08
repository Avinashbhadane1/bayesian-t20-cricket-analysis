import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Ensure the figures directory exists
FIG_DIR = r"D:\Bayesian Project\figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

SA_COLOR = '#006A4E' # Green
WI_COLOR = '#7B0C00' # Maroon
FIG_KWARGS = {'figsize': (10, 6), 'dpi': 150}
SAVE_KWARGS = {'bbox_inches': 'tight'}

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_innings_data(innings_json, team_name):
    overs = innings_json.get('overs', [])
    
    ball_data = []
    
    for over in overs:
        over_num = over['over']
        for ball_idx, delivery in enumerate(over['deliveries']):
            runs_batter = delivery['runs']['batter']
            runs_extras = delivery['runs']['extras']
            runs_total = delivery['runs']['total']
            is_wicket = 1 if 'wickets' in delivery else 0
            
            # check for 4 or 6
            is_four = 1 if runs_batter == 4 else 0
            is_six = 1 if runs_batter == 6 else 0
            
            ball_data.append({
                'over': over_num,
                'ball': ball_idx + 1,
                'runs_batter': runs_batter,
                'runs_extras': runs_extras,
                'runs_total': runs_total,
                'is_wicket': is_wicket,
                'is_four': is_four,
                'is_six': is_six
            })
            
    df = pd.DataFrame(ball_data)
    df['cumulative_runs'] = df['runs_total'].cumsum()
    df['cumulative_wickets'] = df['is_wicket'].cumsum()
    return df

def analyze_match():
    print("="*80)
    print("BAYESIAN CRICKET ANALYSIS: SA vs WI, 2nd T20I (2015)")
    print("="*80)
    
    # 1. Data Loading
    data = load_data(r"D:\Bayesian Project\722337.json")
    innings = data['innings']
    sa_df = extract_innings_data(innings[0], 'South Africa')
    wi_df = extract_innings_data(innings[1], 'West Indies')
    
    # Runs per over calculation
    sa_rpo = sa_df.groupby('over')['runs_total'].sum()
    wi_rpo = wi_df.groupby('over')['runs_total'].sum()
    
    sa_wpo = sa_df.groupby('over')['is_wicket'].sum()
    wi_wpo = wi_df.groupby('over')['is_wicket'].sum()

    print(f"\n[1] Data Loaded.")
    print(f"South Africa Total: {sa_df['runs_total'].sum()}/{sa_df['is_wicket'].sum()} in {len(sa_rpo)} overs")
    print(f"West Indies Total: {wi_df['runs_total'].sum()}/{wi_df['is_wicket'].sum()} in {len(wi_rpo)} overs")

    # --- Fig 1: Runs per over ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    x_sa = sa_rpo.index
    x_wi = wi_rpo.index
    width = 0.4
    ax.bar(x_sa - width/2, sa_rpo.values, width, label='SA', color=SA_COLOR, alpha=0.8)
    ax.bar(x_wi + width/2, wi_rpo.values, width, label='WI', color=WI_COLOR, alpha=0.8)
    ax.set_xlabel('Over')
    ax.set_ylabel('Runs Scored')
    ax.set_title('Runs Per Over: SA vs WI')
    ax.set_xticks(range(20))
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_runs_per_over.png'), **SAVE_KWARGS)
    plt.close()

    # --- Fig 2: Cumulative Runs ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    ax.plot(range(1, len(sa_rpo)+1), sa_rpo.cumsum(), marker='o', color=SA_COLOR, label='SA', linewidth=2)
    ax.plot(range(1, len(wi_rpo)+1), wi_rpo.cumsum(), marker='s', color=WI_COLOR, label='WI', linewidth=2)
    ax.set_xlabel('Over')
    ax.set_ylabel('Cumulative Runs')
    ax.set_title('Cumulative Run Progression (Worm Chart)')
    ax.set_xticks(range(1, 21))
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_cumulative_runs.png'), **SAVE_KWARGS)
    plt.close()

    # 2. Distribution of Runs Per Over (Poisson Model)
    all_rpo = pd.concat([sa_rpo, wi_rpo])
    lambda_mle = all_rpo.mean()
    print("\n[2] POISSON MODEL FOR RUNS PER OVER")
    print(f"Mean runs per over (λ) = {lambda_mle:.2f}")
    
    # --- Fig 3: Poisson Fit ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    max_runs = all_rpo.max()
    x_vals = np.arange(0, max_runs + 5)
    poisson_pmf = stats.poisson.pmf(x_vals, lambda_mle)
    
    ax.hist(all_rpo, bins=np.arange(-0.5, max_runs+1.5, 1), density=True, alpha=0.6, color='steelblue', label='Observed (Both Innings)')
    ax.plot(x_vals, poisson_pmf, 'ko-', label=f'Fitted Poisson (λ={lambda_mle:.2f})')
    ax.set_xlabel('Runs in an Over')
    ax.set_ylabel('Probability')
    ax.set_title('Distribution of Runs Per Over with Poisson Fit')
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_poisson_fit.png'), **SAVE_KWARGS)
    plt.close()

    # 3. Probability of Scoring More Than 180 Runs
    # Total score X ~ Normal approx or Poisson sum
    # If 1 over ~ Poisson(lambda), 20 overs ~ Poisson(20 * lambda)
    total_lambda = 20 * lambda_mle
    p_gt_180 = 1 - stats.poisson.cdf(180, total_lambda)
    
    print("\n[3] PROBABILITY OF SCORE > 180 (Using Poisson Approximation)")
    print(f"Total innings λ (20 overs) = {total_lambda:.2f}")
    print(f"P(X > 180) = {p_gt_180:.4f} ({p_gt_180*100:.2f}%)")

    # 4. Probability of Losing a Wicket in the Next Over (Beta-Binomial)
    # We model wicket in an over as Binomial(1, p) or simply p for losing at least one wicket.
    # Let's count how many overs had at least one wicket across the match.
    wpo_binary = (all_rpo.index.map(lambda i: sa_wpo.get(i, 0) > 0).tolist() + 
                  wi_rpo.index.map(lambda i: wi_wpo.get(i, 0) > 0).tolist())
    
    wpo_binary = np.array(all_rpo.index.map(lambda x: 0)) # reset
    overs_with_wickets_sa = (sa_wpo > 0).sum()
    overs_with_wickets_wi = (wi_wpo > 0).sum()
    total_overs = len(sa_rpo) + len(wi_rpo)
    total_overs_with_wickets = overs_with_wickets_sa + overs_with_wickets_wi
    
    prior_a, prior_b = 1, 1
    post_a = prior_a + total_overs_with_wickets
    post_b = prior_b + total_overs - total_overs_with_wickets
    
    print("\n[4] BAYESIAN WICKET PROBABILITY (Beta-Binomial)")
    print(f"Prior: Beta({prior_a}, {prior_b})")
    print(f"Likelihood: {total_overs_with_wickets} overs with wickets out of {total_overs} total overs")
    print(f"Posterior: Beta({post_a}, {post_b})")
    print(f"Expected Probability of Wicket in an over = {post_a/(post_a+post_b):.4f}")

    # --- Fig 4: Wicket Probability Posterior ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    x = np.linspace(0, 1, 500)
    ax.plot(x, stats.beta.pdf(x, prior_a, prior_b), 'r--', label=f'Prior Beta({prior_a},{prior_b})')
    ax.plot(x, stats.beta.pdf(x, post_a, post_b), 'b-', linewidth=2, label=f'Posterior Beta({post_a},{post_b})')
    ax.set_xlabel('Probability of a Wicket in an Over ($p$)')
    ax.set_ylabel('Density')
    ax.set_title('Beta Prior vs Posterior for Wicket Probability')
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_wicket_probability_posterior.png'), **SAVE_KWARGS)
    plt.close()

    # 5. Expected Number of Runs (Gamma-Poisson)
    # Update after observing first 6 overs (Powerplay) for SA
    pp_overs = sa_rpo.head(6)
    runs_in_pp = pp_overs.sum()
    n_pp = len(pp_overs)
    
    prior_alpha, prior_beta = 8, 1 # Weakly informative, expecting ~8 runs/over
    post_alpha = prior_alpha + runs_in_pp
    post_beta = prior_beta + n_pp
    
    print("\n[5] EXPECTED NUMBER OF RUNS (Gamma-Poisson)")
    print(f"Prior: Gamma({prior_alpha}, {prior_beta})")
    print(f"Observed in Powerplay (SA): {runs_in_pp} runs in {n_pp} overs")
    print(f"Posterior: Gamma({post_alpha}, {post_beta})")
    print(f"Updated expected run rate λ = {post_alpha/post_beta:.2f} runs/over")

    # --- Fig 5: Expected Runs Posterior ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    x = np.linspace(0, 25, 500)
    ax.plot(x, stats.gamma.pdf(x, a=prior_alpha, scale=1/prior_beta), 'r--', label=f'Prior Gamma({prior_alpha},{prior_beta})')
    # Likelihood scaled for visualization
    likelihood = stats.poisson.pmf(runs_in_pp, mu=x*n_pp)
    ax.plot(x, likelihood/np.max(likelihood) * np.max(stats.gamma.pdf(x, a=post_alpha, scale=1/post_beta)), 'g:', label='Likelihood (scaled)')
    ax.plot(x, stats.gamma.pdf(x, a=post_alpha, scale=1/post_beta), 'b-', linewidth=2, label=f'Posterior Gamma({post_alpha},{post_beta})')
    ax.set_xlabel(r'Expected Runs Per Over ($\lambda$)')
    ax.set_ylabel('Density')
    ax.set_title('Gamma Prior vs Posterior for Run Rate (After Powerplay)')
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_expected_runs_posterior.png'), **SAVE_KWARGS)
    plt.close()

    # 6. Bayesian Win Probability Updating
    print("\n[6] BAYESIAN WIN PROBABILITY UPDATING")
    
    # 1st Innings (SA Batting)
    # Target par score = 170. We model logistic probability based on projected score.
    # P(Win) = 1 / (1 + exp(-(proj_score - par_score) / scale))
    par_score = 170
    scale = 20
    sa_proj = []
    sa_win_prob = []
    prior_sa_win = 0.55
    
    sa_win_prob.append(prior_sa_win)
    current_runs = 0
    
    for i, r in enumerate(sa_rpo):
        current_runs += r
        overs_completed = i + 1
        current_rr = current_runs / overs_completed
        proj = current_rr * 20
        sa_proj.append(proj)
        
        # Bayesian logic using simple logistic based on projection vs par
        # This provides a dynamic likelihood updating.
        logit = (proj - par_score) / scale
        p_win_given_proj = 1 / (1 + np.exp(-logit))
        
        # Smooth with prior initially
        weight = overs_completed / 20.0
        updated_prob = (1 - weight) * prior_sa_win + weight * p_win_given_proj
        sa_win_prob.append(updated_prob)
        
        if overs_completed == 10:
            print(f"-> 1st Innings, Over 10: SA score {current_runs}. Win Prob = {updated_prob:.4f}")

    # 2nd Innings (WI Chasing)
    # Target = 232
    target = 232
    wi_win_prob = []
    # Prior based on massive target
    prior_wi_win = 0.10
    wi_win_prob.append(prior_wi_win)
    
    current_runs_wi = 0
    current_wickets_wi = 0
    
    for i, r in enumerate(wi_rpo):
        current_runs_wi += r
        w = wi_wpo.iloc[i] if i < len(wi_wpo) else 0
        current_wickets_wi += w
        
        overs_completed = i + 1
        runs_req = target - current_runs_wi
        overs_left = 20 - overs_completed
        
        if overs_left > 0:
            req_rr = runs_req / overs_left
            curr_rr = current_runs_wi / overs_completed
            
            # Simple heuristic resource model
            # Base probability on difference between Required RR and Current RR, adjusting for wickets
            wkts_in_hand = 10 - current_wickets_wi
            resource_factor = wkts_in_hand / 10.0
            
            logit = (curr_rr - req_rr) * 0.5 + (resource_factor - 0.5) * 5
            p_win_chase = 1 / (1 + np.exp(-logit))
            
            weight = overs_completed / 20.0
            updated_prob = (1 - weight) * prior_wi_win + weight * p_win_chase
        else:
            updated_prob = 1.0 if current_runs_wi >= target else 0.0
            
        wi_win_prob.append(updated_prob)
        
        if overs_completed == 10:
            print(f"-> 2nd Innings, Over 10: WI score {current_runs_wi}/{current_wickets_wi}. Win Prob = {updated_prob:.4f}")

    # --- Fig 6: Win Probability ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    ax.plot(range(0, len(sa_win_prob)), sa_win_prob, marker='o', color=SA_COLOR, label='SA Win Prob (1st Inn)')
    ax.plot(range(20, 20 + len(wi_win_prob)), wi_win_prob, marker='s', color=WI_COLOR, label='WI Win Prob (2nd Inn)')
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(20, color='black', linestyle=':', label='Innings Break')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Overs Completed (Match Progression)')
    ax.set_ylabel('Probability of Winning')
    ax.set_title('Bayesian Over-by-Over Win Probability')
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_win_probability_updating.png'), **SAVE_KWARGS)
    plt.close()

    # --- Fig 7: Scoring Distribution ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    sa_scores = sa_df['runs_batter'].value_counts().sort_index()
    wi_scores = wi_df['runs_batter'].value_counts().sort_index()
    
    x_idx = np.arange(len(set(sa_scores.index).union(wi_scores.index)))
    labels = sorted(list(set(sa_scores.index).union(wi_scores.index)))
    
    sa_counts = [sa_scores.get(l, 0) for l in labels]
    wi_counts = [wi_scores.get(l, 0) for l in labels]
    
    ax.bar(x_idx - 0.2, sa_counts, 0.4, color=SA_COLOR, label='SA')
    ax.bar(x_idx + 0.2, wi_counts, 0.4, color=WI_COLOR, label='WI')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Runs from Bat (Per Ball)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Individual Ball Scores')
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig7_scoring_distribution.png'), **SAVE_KWARGS)
    plt.close()

    # --- Fig 8: Boundary Analysis ---
    fig, ax = plt.subplots(**FIG_KWARGS)
    sa_boundaries = sa_df.groupby('over')[['is_four', 'is_six']].sum()
    wi_boundaries = wi_df.groupby('over')[['is_four', 'is_six']].sum()
    
    ax.plot(sa_boundaries.index, sa_boundaries['is_four'] + sa_boundaries['is_six'], 'o-', color=SA_COLOR, label='SA Boundaries')
    ax.plot(wi_boundaries.index, wi_boundaries['is_four'] + wi_boundaries['is_six'], 's-', color=WI_COLOR, label='WI Boundaries')
    ax.set_xlabel('Over')
    ax.set_ylabel('Number of Boundaries (4s and 6s)')
    ax.set_title('Boundaries Per Over')
    ax.set_xticks(range(20))
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'fig8_boundary_analysis.png'), **SAVE_KWARGS)
    plt.close()

    print("\n[7] VISUALIZATIONS GENERATED")
    print(f"All 8 figures successfully saved to {FIG_DIR}")
    print("="*80)

if __name__ == "__main__":
    analyze_match()
