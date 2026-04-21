import matplotlib.pyplot as plt
from env import StartupEnv
from baseline import choose_action as naive_action
from inference import choose_preferred_action as trained_action

def run_agent(policy_func, task="survival"):
    env = StartupEnv(seed=42)
    state = env.state().model_dump()
    cash_history = [state["cash"]]
    users_history = [state["users"]]
    reward_history = [0.0]
    cumulative_reward = 0.0
    
    done = False
    while not done:
        action = policy_func(task, state)
        res = env.step(action)
        done = res["done"]
        state = res["state"]
        cash_history.append(state["cash"])
        users_history.append(state["users"])
        cumulative_reward += res["reward"]
        reward_history.append(cumulative_reward)
        
    return cash_history, users_history, reward_history

if __name__ == "__main__":
    print("Running naive baseline...")
    naive_cash, naive_users, naive_reward = run_agent(naive_action, "survival")
    
    print("Running trained heuristic...")
    trained_cash, trained_users, trained_reward = run_agent(trained_action, "survival")
    
    plt.figure(figsize=(12, 5))
    
    # Plot Reward
    plt.subplot(1, 2, 1)
    plt.plot(naive_reward, label="Naive Policy", color="#ff3333", linestyle="--", linewidth=2)
    plt.plot(trained_reward, label="Trained Policy", color="#00c853", linewidth=2)
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Cumulative Reward (Sparse)", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Reward Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot Users
    plt.subplot(1, 2, 2)
    plt.plot(naive_users, label="Naive Policy (Bankrupt)", color="#ff3333", linestyle="--", linewidth=2)
    plt.plot(trained_users, label="Trained Policy", color="#00c853", linewidth=2)
    plt.title("User Growth Over Time", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Active Users", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("comparison_chart.png", dpi=300)
    print("Saved comparison_chart.png successfully!")
