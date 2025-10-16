import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# --- Configuration ---
NUM_USERS = 1000
NUM_ITEMS = 500
TIME_STEPS = 10
BIAS_FACTOR = 0.6  # How much the initial algorithm favors Group A

# --- 1. Data Generation and Setup ---
print("Setting up initial data...")

# User Data: 'group' (0 or 1), 'activity' (initial engagement level)
users_df = pd.DataFrame({
    'user_id': range(NUM_USERS),
    'group': np.random.choice([0, 1], size=NUM_USERS, p=[0.3, 0.7]), # 30% Group 0 (A), 70% Group 1 (B)
    'activity': np.random.rand(NUM_USERS) * 0.5 + 0.5 # Initial engagement (0.5 to 1.0)
})

# Item Data: 'quality' (relevance/attractiveness)
items_df = pd.DataFrame({
    'item_id': range(NUM_ITEMS),
    'quality': np.random.rand(NUM_ITEMS)
})

# Initialize exposure counter for each item
items_df['exposure_count'] = 0

# --- 2. Bias Metric Operationalization (Disparity in Exposure) ---
def calculate_disparity(users_df: pd.DataFrame, items_df: pd.DataFrame) -> float:
    """Calculates the disparity in total exposure between Group A (0) and Group B (1)."""

    # Merge user groups with their activity (which will be used as a proxy for *generated* content/feedback)
    user_exposure = users_df.groupby('group')['activity'].sum()

    exposure_A = user_exposure.get(0, 0)
    exposure_B = user_exposure.get(1, 0)

    # Disparity: Absolute difference relative to the total exposure
    total_exposure = exposure_A + exposure_B
    if total_exposure == 0:
        return 0.0

    # We'll use a simple ratio for demonstration: (Exposure_A / Exposure_B)
    # A value > 1 means Group A receives disproportionately more exposure.
    if exposure_B == 0: return float('inf')
    disparity_ratio = exposure_A / exposure_B

    return disparity_ratio

# --- 3. The Core Simulation (Feedback Loop) ---

def simulate_recommendation_round(users_df: pd.DataFrame, items_df: pd.DataFrame, bias_factor: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates one round of recommendation, user interaction, and feedback.
    1. Recommendation: biased towards Group A.
    2. Interaction: Group A users interact more due to higher recommendations.
    3. Feedback: Interaction updates the next round's 'activity' (proxy for generating content/feedback).
    """

    # 1. Recommendation (Based on a combination of item quality and algorithmic bias)
    # Higher bias_factor means Group A items are *more likely* to be recommended.
    
    recommended_items = []
    
    for _, user in users_df.iterrows():
        is_group_A = user['group'] == 0
        
        # Calculate a recommendation score: quality + (bias if Group A)
        scores = items_df['quality'].copy()
        
        # Apply algorithmic bias: boost scores for Group A (simulated here by biasing towards items associated with Group A's activity)
        # In a real model, this would be based on training data that reflects past bias.
        if is_group_A:
            # Simple, non-rigorous example of bias: Group A gets a score boost
            scores += (user['activity'] * bias_factor) 
        
        # Select one item (simplification)
        recommended_item_id = scores.idxmax()
        recommended_items.append((user['user_id'], recommended_item_id))
        
        # Update item exposure count
        items_df.loc[recommended_item_id, 'exposure_count'] += 1

    # 2. User Interaction (Interaction is proportional to their current activity level)
    # The users who *received* recommendations interact and generate 'feedback'.
    
    new_activity = users_df['activity'].copy()
    
    for user_id, item_id in recommended_items:
        user_index = users_df[users_df['user_id'] == user_id].index[0]
        
        # Interaction success/feedback (Higher activity = higher chance of successful interaction/feedback)
        if np.random.rand() < users_df.loc[user_index, 'activity']:
            # 3. Feedback: Successful interaction increases the user's 'activity' for the next round
            new_activity.loc[user_index] += 0.05 # Amplify activity/feedback
            
    users_df['activity'] = new_activity.clip(max=1.5) # Cap activity for stability

    return users_df, items_df

# --- 4. Main Experiment Loop ---
print("\nStarting feedback loop simulation...")

disparity_history = []

for t in range(TIME_STEPS):
    users_df, items_df = simulate_recommendation_round(users_df, items_df, BIAS_FACTOR)
    current_disparity = calculate_disparity(users_df, items_df)
    disparity_history.append(current_disparity)

    print(f"Time Step {t+1}: Disparity Ratio (Group A / Group B Activity) = {current_disparity:.4f}")

# --- 5. Results and Analysis ---
print("\n--- Simulation Results ---")
initial_disparity = disparity_history[0]
final_disparity = disparity_history[-1]

print(f"Initial Disparity (Step 1): {initial_disparity:.4f}")
print(f"Final Disparity (Step {TIME_STEPS}): {final_disparity:.4f}")

# Calculate the amplification percentage
if initial_disparity > 0:
    amplification_percent = ((final_disparity - initial_disparity) / initial_disparity) * 100
    print(f"Bias Amplification due to Feedback Loop: +{amplification_percent:.2f}%")
else:
    print("Cannot calculate amplification percentage.")

# Visualization (Requires matplotlib, which isn't installed here, but the code would look like this):
# import matplotlib.pyplot as plt
# plt.plot(range(1, TIME_STEPS + 1), disparity_history)
# plt.title("Bias Evolution Over Time")
# plt.xlabel("Time Step (Iteration)")
# plt.ylabel("Disparity Ratio (A/B)")
# plt.show()
